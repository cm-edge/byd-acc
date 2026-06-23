"""
PTC (Predictive Temporal Consistency) estimator — self-supervised, no model weights.

Three scores per frame transition (t-1 → t):
  PTC_Det : confidence-weighted Jaccard similarity between consecutive detections
  PTC_Loc : motion-prediction DIoU-like localization consistency
  PTC_H   : geometric mean of PTC_Det and PTC_Loc  (main routing signal)

Low PTC_H  →  weak model temporally inconsistent  →  offload to strong model.
High PTC_H →  weak model consistent               →  keep weak model.

Detections format expected:
  List of (x, y, w, h, conf, class_str)  — one list per frame (xywh pixel coords)

Usage:
    from ptc_inference import PTCEstimator
    est   = PTCEstimator()
    score = est.update(detections_t)   # call every frame → returns dict or None
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

Det  = Tuple[float, float, float, float, float, str]  # x,y,w,h,conf,cls
BBox = Tuple[float, float, float, float]               # x,y,w,h

IOU_THR     = 0.5
DIOU_ALPHA  = 0.7


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _iou(b1: BBox, b2: BBox) -> float:
    x1, y1, w1, h1 = b1;  x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1,x2), max(y1,y2)
    ix2, iy2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    union = max(0.0,w1)*max(0.0,h1) + max(0.0,w2)*max(0.0,h2) - inter
    return inter/union if union>0 else 0.0


def _center(b: BBox) -> Tuple[float, float]:
    return b[0]+b[2]/2, b[1]+b[3]/2


def _from_center(cx, cy, w, h) -> BBox:
    return cx-w/2, cy-h/2, w, h


def _diou_like(b_pred: BBox, b_curr: BBox, alpha: float = DIOU_ALPHA) -> float:
    iou = _iou(b_pred, b_curr)
    cx1, cy1 = _center(b_pred);  cx2, cy2 = _center(b_curr)
    d2 = (cx1-cx2)**2 + (cy1-cy2)**2
    ex1, ey1 = min(b_pred[0],b_curr[0]), min(b_pred[1],b_curr[1])
    ex2 = max(b_pred[0]+b_pred[2], b_curr[0]+b_curr[2])
    ey2 = max(b_pred[1]+b_pred[3], b_curr[1]+b_curr[3])
    c2  = (ex2-ex1)**2 + (ey2-ey1)**2 + 1e-12
    cs  = float(np.exp(-d2/c2))
    return float(np.clip(iou**alpha * cs**(1-alpha), 0.0, 1.0))


def _predict_box(pp: BBox, p: BBox) -> BBox:
    cx_k, cy_k = _center(pp);  cx_i, cy_i = _center(p)
    eps = 1e-12
    pw  = float(np.exp(np.log(p[2]+eps)*2 - np.log(pp[2]+eps)))
    ph  = float(np.exp(np.log(p[3]+eps)*2 - np.log(pp[3]+eps)))
    return _from_center(cx_i+(cx_i-cx_k), cy_i+(cy_i-cy_k), max(1e-6,pw), max(1e-6,ph))


# ---------------------------------------------------------------------------
# Greedy IoU matching
# ---------------------------------------------------------------------------
def _greedy_match(a: List[Det], b: List[Det]) -> List[Tuple[int,int,float]]:
    if not a or not b:
        return []
    mat = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, d1 in enumerate(a):
        for j, d2 in enumerate(b):
            if d1[5] == d2[5]:
                mat[i,j] = _iou(d1[:4], d2[:4])
    matches, tmp = [], mat.copy()
    while True:
        idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        if tmp[idx] < IOU_THR:
            break
        matches.append((int(idx[0]), int(idx[1]), float(tmp[idx])))
        tmp[idx[0],:] = -1;  tmp[:,idx[1]] = -1
    return matches


# ---------------------------------------------------------------------------
# Raw PTC scores for one frame transition
# ---------------------------------------------------------------------------
def ptc_det_raw(prev: List[Det], curr: List[Det]) -> float:
    if not prev and not curr:
        return 1.0
    if not prev or not curr:
        return 0.0
    matches  = _greedy_match(prev, curr)
    if not matches:
        return 0.0
    prev_sum = sum(float(d[4]) for d in prev)
    curr_sum = sum(float(d[4]) for d in curr)
    matched  = sum(min(float(prev[i][4]), float(curr[j][4])) for i,j,_ in matches)
    denom    = prev_sum + curr_sum - matched
    return float(matched/denom) if denom > 0 else 0.0


def ptc_loc_raw(prev: List[Det], curr: List[Det],
                prevprev: List[Det]) -> float:
    if not prev and not curr:
        return 1.0
    if not prev or not curr:
        return 0.0
    matches = _greedy_match(prev, curr)
    if not matches:
        return 0.0
    vals = []
    for pi, ci, match_iou in matches:
        if prevprev:
            best_k = max(range(len(prevprev)),
                         key=lambda k: _iou(prevprev[k][:4], prev[pi][:4]))
            if _iou(prevprev[best_k][:4], prev[pi][:4]) > 0:
                pred = _predict_box(prevprev[best_k][:4], prev[pi][:4])
                vals.append(_diou_like(pred, curr[ci][:4]))
                continue
        vals.append(float(match_iou))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Stateful online estimator
# ---------------------------------------------------------------------------
class PTCEstimator:
    """
    Online PTC estimator: call update() once per frame.

    Parameters
    ----------
    smooth_window : centred moving-average window (set to 1 to disable smoothing)
    iof_thr       : ignored-region filter threshold
    conf_thr      : detection confidence threshold

    update() returns None for the first frame (no previous frame yet),
    and a dict {'ptc_det', 'ptc_loc', 'ptc_h'} from the second frame onwards.
    Scores are NOT smoothed in online mode (smoothing requires a full sequence).
    """

    def __init__(self, conf_thr: float = 0.5, iof_thr: float = 0.5):
        self.conf_thr   = conf_thr
        self.iof_thr    = iof_thr
        self._prev      : List[Det] = []
        self._prevprev  : List[Det] = []
        self._frame_idx : int       = 0

    def reset(self):
        self._prev     = []
        self._prevprev = []
        self._frame_idx = 0

    def _filter(self, dets: List[Det],
                ignored: Optional[List[BBox]] = None) -> List[Det]:
        out = []
        for d in dets:
            x, y, w, h, conf, cls = d
            if conf < self.conf_thr:
                continue
            if ignored:
                iof = 0.0
                for ib in ignored:
                    ix, iy, iw, ih = ib
                    inter = max(0.0,min(x+w,ix+iw)-max(x,ix)) * \
                            max(0.0,min(y+h,iy+ih)-max(y,iy))
                    area  = max(0.0,w)*max(0.0,h)
                    iof   = max(iof, inter/area if area>0 else 0.0)
                if iof > self.iof_thr:
                    continue
            out.append(d)
        return out

    def update(self, dets: List[Det],
               ignored: Optional[List[BBox]] = None) -> Optional[Dict[str, float]]:
        """
        Feed detections for the current frame.

        Parameters
        ----------
        dets    : list of (x,y,w,h,conf,class_str) for this frame (xywh pixels)
        ignored : list of (x,y,w,h) ignored-region boxes, or None

        Returns
        -------
        dict with ptc_det, ptc_loc, ptc_h  — or None on the first frame
        """
        curr = self._filter(dets, ignored)
        self._frame_idx += 1

        if self._frame_idx == 1:
            self._prev = curr
            return None

        det_score = ptc_det_raw(self._prev, curr)
        loc_score = ptc_loc_raw(self._prev, curr, self._prevprev)
        h_score   = float(np.sqrt(max(0.0, det_score) * max(0.0, loc_score)))

        self._prevprev = self._prev
        self._prev     = curr

        return {
            'ptc_det': float(np.clip(det_score, 0.0, 1.0)),
            'ptc_loc': float(np.clip(loc_score, 0.0, 1.0)),
            'ptc_h':   float(np.clip(h_score,   0.0, 1.0)),
        }


# ---------------------------------------------------------------------------
# Batch helper: score a whole video sequence (with smoothing)
# ---------------------------------------------------------------------------
def score_sequence(dets_by_frame: Dict[int, List[Det]],
                   frame_ids: List[int],
                   smooth_window: int = 9) -> List[Dict]:
    """
    Score a complete sequence (offline, with centred-MA smoothing).

    Parameters
    ----------
    dets_by_frame : {frame_id: [det, ...]}
    frame_ids     : ordered list of frame ids
    smooth_window : MA window (must be odd; 1 = no smoothing)

    Returns
    -------
    list of dicts [{frame_id, ptc_det, ptc_loc, ptc_h}, ...]
    """
    T = len(frame_ids)
    raw_det = np.ones(T, np.float32)
    raw_loc = np.ones(T, np.float32)

    for t in range(1, T):
        prev     = dets_by_frame.get(frame_ids[t-1], [])
        curr     = dets_by_frame.get(frame_ids[t],   [])
        prevprev = dets_by_frame.get(frame_ids[t-2], []) if t >= 2 else []
        raw_det[t] = ptc_det_raw(prev, curr)
        raw_loc[t] = ptc_loc_raw(prev, curr, prevprev)

    def _ma(x, k):
        if k <= 1:
            return x.copy()
        half = k // 2
        y = np.empty_like(x)
        for i in range(len(x)):
            y[i] = np.mean(x[max(0,i-half):min(len(x),i+half+1)])
        return y

    ptc_det = np.clip(_ma(raw_det, smooth_window), 0.0, 1.0)
    ptc_loc = np.clip(_ma(raw_loc, smooth_window), 0.0, 1.0)
    ptc_h   = np.sqrt(ptc_det * ptc_loc)

    return [{'frame_id': fid, 'ptc_det': float(ptc_det[i]),
             'ptc_loc': float(ptc_loc[i]), 'ptc_h': float(ptc_h[i])}
            for i, fid in enumerate(frame_ids)]
