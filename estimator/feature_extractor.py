"""
Online stateful feature extractor for XGBoost routing predictor.

Converts raw FasterRCNN detections (per frame) into the 28-dimensional feature
vector expected by the XGBoost model, maintaining a lightweight IoU tracker
internally to compute tracker-derived features without any external tracker.

FasterRCNN output format (torchvision):
    boxes  : (N, 4) float32 xyxy pixel coords
    scores : (N,)   float32 confidence in [0, 1]
    labels : (N,)   int64   COCO category ids

Usage
-----
    from feature_extractor import FrameFeatureExtractor
    extractor = FrameFeatureExtractor(image_width=1920, image_height=1080)

    # called once per frame with raw torchvision detector output
    feat_dict = extractor.update(boxes, scores, labels)
    # feat_dict is a dict[str, float] ready for the sliding window
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BBox4 = Tuple[float, float, float, float]   # x, y, w, h  (pixel)
Track = Tuple[float, float, float, float, float, int]  # x,y,w,h,conf,track_id


# ---------------------------------------------------------------------------
# Simple greedy IoU tracker
# ---------------------------------------------------------------------------
class _GreedyTracker:
    """
    Minimal IoU-based tracker: assigns stable integer track IDs to detections
    across consecutive frames.  No Kalman filter; just greedy IoU matching.
    Used only to compute tracker-derived features — not for tracking evaluation.
    """

    def __init__(self, iou_thr: float = 0.3):
        self._iou_thr    = iou_thr
        self._next_id    = 0
        self._prev_xywh : List[BBox4] = []
        self._prev_ids  : List[int]   = []

    def update(self, xywh_list: List[BBox4]) -> List[int]:
        """Match detections to previous frame; return track_id list (same order)."""
        if not xywh_list:
            self._prev_xywh = []
            self._prev_ids  = []
            return []

        if not self._prev_xywh:
            ids = [self._next_id + i for i in range(len(xywh_list))]
            self._next_id += len(xywh_list)
            self._prev_xywh = list(xywh_list)
            self._prev_ids  = ids
            return ids

        # IoU cost matrix
        n, m = len(xywh_list), len(self._prev_xywh)
        cost = np.zeros((n, m), dtype=np.float32)
        for i, a in enumerate(xywh_list):
            for j, b in enumerate(self._prev_xywh):
                cost[i, j] = _iou_xywh(a, b)

        assigned_curr : Dict[int, int] = {}   # curr_idx → prev_idx
        tmp = cost.copy()
        while True:
            idx = np.unravel_index(np.argmax(tmp), tmp.shape)
            if tmp[idx] < self._iou_thr:
                break
            ci, pi = int(idx[0]), int(idx[1])
            assigned_curr[ci] = pi
            tmp[ci, :] = -1
            tmp[:, pi] = -1

        ids = []
        for i in range(n):
            if i in assigned_curr:
                ids.append(self._prev_ids[assigned_curr[i]])
            else:
                ids.append(self._next_id)
                self._next_id += 1

        self._prev_xywh = list(xywh_list)
        self._prev_ids  = ids
        return ids

    def reset(self):
        self._next_id   = 0
        self._prev_xywh = []
        self._prev_ids  = []


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou_xywh(a: BBox4, b: BBox4) -> float:
    ax, ay, aw, ah = a;  bx, by, bw, bh = b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------
class FrameFeatureExtractor:
    """
    Stateful online extractor.  Call update() once per frame in order.
    Returns a dict[str, float] with the 28 features the XGBoost model expects.
    """

    FEATURE_COLUMNS = [
        'bot0_area_log', 'bot0_aspect_ratio', 'bot0_track_changed',
        'det_count', 'delta_det_count',
        'conf_min', 'conf_max', 'conf_median', 'conf_entropy',
        'conf_top_k_gap_k2', 'conf_top_k_gap_k3', 'conf_top_k_gap_k5',
        'small_box_ratio', 'ratio_outlier_count', 'deformed_box_count',
        'intra_iou_mean', 'intra_iou_max',
        'track_continuity_ratio', 'centroid_jitter_mean', 'centroid_jitter_max',
        'new_track_ratio', 'lost_track_count',
        'center_vanishing_ratio', 'high_conf_vanishing_ratio', 'new_high_conf_ratio',
        'ghost_prediction_error', 'track_age_imbalance', 'occlusion_risk_count',
    ]

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 conf_thr: float = 0.1,
                 history_window: int = 5,
                 iou_tracker_thr: float = 0.3):
        """
        Parameters
        ----------
        image_width, image_height : frame resolution (pixels)
        conf_thr    : minimum detection confidence to keep (matches offline pipeline default)
        history_window : track-continuity look-back (frames)
        iou_tracker_thr : greedy IoU threshold for track assignment
        """
        self.image_width   = image_width
        self.image_height  = image_height
        self.conf_thr      = conf_thr
        self.history_window = history_window

        self._tracker      = _GreedyTracker(iou_thr=iou_tracker_thr)
        self._track_history: Dict[int, List[Tuple[float, float]]] = {}  # id → [(cx,cy), ...]
        self._prev_dets    : List[Track] = []     # from previous frame
        self._frame_idx    = 0

    def reset(self):
        """Call at the start of a new video sequence."""
        self._tracker.reset()
        self._track_history.clear()
        self._prev_dets = []
        self._frame_idx = 0

    # ------------------------------------------------------------------
    def update(self,
               boxes_xyxy : np.ndarray,
               scores     : np.ndarray,
               labels     : Optional[np.ndarray] = None,
               vehicle_ids: Optional[set]        = None,
               ) -> Dict[str, float]:
        """
        Process one frame and return feature dict.

        Parameters
        ----------
        boxes_xyxy  : (N,4) float32 array of xyxy pixel boxes (torchvision format)
        scores      : (N,)  float32 confidence scores
        labels      : (N,)  int array of COCO label ids (optional)
                      if vehicle_ids is set, only those labels are kept
        vehicle_ids : set of COCO label ids to keep, e.g. {3,6,8} for car/bus/truck
                      None = keep all labels

        Returns
        -------
        dict[str, float] — 28 features, always present, NaN-free
        """
        self._frame_idx += 1

        # --- filter by confidence and class ---
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        scores     = np.asarray(scores,     dtype=np.float32)

        if len(boxes_xyxy) == 0:
            keep = np.array([], dtype=np.int64)
        else:
            keep = scores >= self.conf_thr
            if vehicle_ids is not None and labels is not None:
                labels = np.asarray(labels)
                keep = keep & np.array([int(l) in vehicle_ids for l in labels])
            keep = np.where(keep)[0]

        xywh_list : List[BBox4] = []
        conf_list : List[float] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_xyxy[i]
            xywh_list.append((float(x1), float(y1),
                               float(x2 - x1), float(y2 - y1)))
            conf_list.append(float(scores[i]))

        # --- assign track IDs ---
        track_ids = self._tracker.update(xywh_list)

        # Build detection tuples: (x, y, w, h, conf, track_id)
        curr_dets: List[Track] = [
            (xywh_list[i][0], xywh_list[i][1],
             xywh_list[i][2], xywh_list[i][3],
             conf_list[i], track_ids[i])
            for i in range(len(xywh_list))
        ]

        # --- compute features ---
        feat = self._compute(curr_dets, self._prev_dets, self._track_history)

        # --- update state ---
        for tid, (x, y, w, h) in zip(track_ids, xywh_list):
            cx, cy = x + w / 2, y + h / 2
            if tid not in self._track_history:
                self._track_history[tid] = []
            self._track_history[tid].append((cx, cy))
            if len(self._track_history[tid]) > self.history_window + 2:
                self._track_history[tid] = self._track_history[tid][-(self.history_window + 2):]

        self._prev_dets = curr_dets
        return feat

    # ------------------------------------------------------------------
    def _compute(self,
                 dets     : List[Track],
                 prev_dets: List[Track],
                 history  : Dict[int, List[Tuple[float, float]]],
                 ) -> Dict[str, float]:
        W, H = self.image_width, self.image_height

        if not dets:
            return {c: 0.0 for c in self.FEATURE_COLUMNS}

        confs  = np.array([d[4] for d in dets], dtype=np.float32)
        areas  = np.array([d[2] * d[3] for d in dets], dtype=np.float32)
        widths = np.array([d[2] for d in dets], dtype=np.float32)
        heights= np.array([d[3] for d in dets], dtype=np.float32)
        aspect = widths / (heights + 1e-6)
        n      = len(dets)

        feat: Dict[str, float] = {}

        # ---- bot0: lowest-confidence detection ----
        bot_idx = int(np.argmin(confs))
        bd = dets[bot_idx]
        feat['bot0_area_log']      = float(np.log1p(bd[2] * bd[3]))
        feat['bot0_aspect_ratio']  = float(bd[2] / (bd[3] + 1e-6))
        prev_ids = {d[5] for d in prev_dets}
        feat['bot0_track_changed'] = 0.0 if (prev_dets and bd[5] in prev_ids) else 1.0

        # ---- confidence stats ----
        feat['det_count']          = float(n)
        feat['delta_det_count']    = float(n - len(prev_dets)) if prev_dets else 0.0
        feat['conf_min']           = float(np.min(confs))
        feat['conf_max']           = float(np.max(confs))
        feat['conf_median']        = float(np.median(confs))
        s_desc = np.sort(confs)[::-1]
        feat['conf_top_k_gap_k2']  = float(s_desc[0] - s_desc[1]) if n >= 2 else 0.0
        feat['conf_top_k_gap_k3']  = float(s_desc[0] - s_desc[2]) if n >= 3 else 0.0
        feat['conf_top_k_gap_k5']  = float(s_desc[0] - s_desc[4]) if n >= 5 else 0.0
        cs = confs / (confs.sum() + 1e-10)
        feat['conf_entropy']       = float(-np.sum(cs * np.log(cs + 1e-10)))

        # ---- geometry ----
        small_thr = W * H * 0.01
        feat['small_box_ratio']     = float(np.sum(areas < small_thr) / n)
        feat['ratio_outlier_count'] = float(np.sum((aspect < 0.5) | (aspect > 4.0)))
        feat['deformed_box_count']  = float(np.sum(aspect > 3.0))

        # intra-frame IoU — vectorised to avoid O(N²) Python loop
        if n >= 2:
            xs = np.array([d[0] for d in dets], dtype=np.float32)
            ys = np.array([d[1] for d in dets], dtype=np.float32)
            ws = np.array([d[2] for d in dets], dtype=np.float32)
            hs = np.array([d[3] for d in dets], dtype=np.float32)
            ix1 = np.maximum(xs[:, None], xs[None, :])
            iy1 = np.maximum(ys[:, None], ys[None, :])
            ix2 = np.minimum(xs[:, None] + ws[:, None], xs[None, :] + ws[None, :])
            iy2 = np.minimum(ys[:, None] + hs[:, None], ys[None, :] + hs[None, :])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            union = ws[:, None] * hs[:, None] + ws[None, :] * hs[None, :] - inter
            iou_mat = inter / (union + 1e-6)
            triu = iou_mat[np.triu_indices(n, k=1)]
            feat['intra_iou_mean'] = float(np.mean(triu))
            feat['intra_iou_max']  = float(np.max(triu))
        else:
            feat['intra_iou_mean'] = 0.0
            feat['intra_iou_max']  = 0.0

        # ---- tracker features ----
        curr_ids = {d[5] for d in dets}
        curr_centers = {d[5]: (d[0] + d[2]/2, d[1] + d[3]/2) for d in dets}

        # continuity + new ratio
        continuous, new_count = 0, 0
        for tid in curr_ids:
            hist_len = len(history.get(tid, []))
            if hist_len >= max(1, self.history_window - 2):
                continuous += 1
            elif hist_len == 0:
                new_count  += 1
        feat['track_continuity_ratio'] = continuous / n
        feat['new_track_ratio']        = new_count  / n

        # centroid jitter (std of consecutive displacements)
        jitters = []
        for tid in curr_ids:
            hist = history.get(tid, [])
            if len(hist) >= 2:
                disps = [np.sqrt(((hist[k][0]-hist[k-1][0])/W)**2 +
                                 ((hist[k][1]-hist[k-1][1])/H)**2)
                         for k in range(1, len(hist))]
                if len(disps) >= 2:
                    jitters.append(float(np.std(disps)))
        feat['centroid_jitter_mean'] = float(np.mean(jitters)) if jitters else 0.0
        feat['centroid_jitter_max']  = float(np.max(jitters))  if jitters else 0.0

        # lost tracks
        lost_ids = prev_ids - curr_ids
        feat['lost_track_count'] = float(len(lost_ids))

        # center_vanishing / high_conf_vanishing
        if lost_ids and prev_dets:
            prev_info = {d[5]: d for d in prev_dets}
            mx, my = W * 0.10, H * 0.09
            cv, hv = 0, 0
            for tid in lost_ids:
                if tid not in prev_info:
                    continue
                d = prev_info[tid]
                l, t, w, h, c = d[0], d[1], d[2], d[3], d[4]
                if l > mx and t > my and (W-l-w) > mx and (H-t-h) > my:
                    cv += 1
                if c > 0.8:
                    hv += 1
            total = len(curr_ids | prev_ids)
            feat['center_vanishing_ratio']    = cv / max(total, 1)
            feat['high_conf_vanishing_ratio'] = hv / max(total, 1)
        else:
            feat['center_vanishing_ratio']    = 0.0
            feat['high_conf_vanishing_ratio'] = 0.0

        # new high-conf ratio
        nhc, nt = 0, 0
        for d in dets:
            if len(history.get(d[5], [])) == 0:
                nt += 1
                if d[4] > 0.8:
                    nhc += 1
        feat['new_high_conf_ratio'] = nhc / nt if nt > 0 else 0.0

        # ghost prediction error (linear motion model)
        errors = []
        for tid in curr_ids:
            hist = history.get(tid, [])
            if len(hist) >= 2 and tid in curr_centers:
                vx = hist[-1][0] - hist[-2][0]
                vy = hist[-1][1] - hist[-2][1]
                px, py = hist[-1][0] + vx, hist[-1][1] + vy
                ax, ay = curr_centers[tid]
                errors.append(np.sqrt(((px-ax)/W)**2 + ((py-ay)/H)**2))
        feat['ghost_prediction_error'] = float(np.mean(errors)) if errors else 0.0

        # track age imbalance (fraction of short-lived tracks)
        short = sum(1 for tid in curr_ids if len(history.get(tid, [])) < 2)
        feat['track_age_imbalance'] = short / n

        # occlusion risk (predicted IoU > 0.5 between next-frame positions)
        pred_boxes: Dict[int, BBox4] = {}
        for d in dets:
            tid = d[5]; w, h = d[2], d[3]
            hist = history.get(tid, [])
            if len(hist) >= 2:
                vx = hist[-1][0] - hist[-2][0]
                vy = hist[-1][1] - hist[-2][1]
                px, py = hist[-1][0] + vx, hist[-1][1] + vy
            else:
                px, py = d[0] + w/2, d[1] + h/2
            pred_boxes[tid] = (px - w/2, py - h/2, w, h)
        pids = list(pred_boxes.keys())
        occ = sum(1 for i in range(len(pids))
                  for j in range(i+1, len(pids))
                  if _iou_xywh(pred_boxes[pids[i]], pred_boxes[pids[j]]) > 0.5)
        feat['occlusion_risk_count'] = float(occ)

        return feat


# ---------------------------------------------------------------------------
# Sliding window helper (mirrors DifficultyPredictor.create_temporal_features)
# ---------------------------------------------------------------------------
def build_window_vector(frame_buffer: List[Dict[str, float]],
                        feature_columns: List[str]) -> Optional[np.ndarray]:
    """
    Stack the last window_size feature dicts into a flat 1-D numpy array.

    Parameters
    ----------
    frame_buffer   : deque/list of feature dicts, oldest first
                     must have exactly window_size entries
    feature_columns: ordered list of feature names (from predictor.feature_columns)

    Returns
    -------
    (1, window_size * n_features) float32 array ready for predictor.predict()
    None if buffer is not full yet
    """
    if not frame_buffer:
        return None
    row = np.concatenate([
        [f.get(c, 0.0) for c in feature_columns]
        for f in frame_buffer
    ], dtype=np.float32)
    return row.reshape(1, -1)
