"""
IoUNet quality estimator — inference only.

Architecture: 3-layer MLP (12544 → 1024 → 1024 → 1) that predicts
max IoU between a detected box and any GT box, using RoI-pooled
features from a FasterRCNN-MobileNetV3-Large-FPN backbone.

High iou_net_mean  →  weak model localises well  →  no need to offload.
Low  iou_net_mean  →  weak model unreliable       →  offload to strong model.

Checkpoint key format expected: fc1.weight / fc1.bias / fc2 / fc3
(as saved by the original training script).

Usage:
    from iounet_inference import IoUNetEstimator
    est = IoUNetEstimator("weights/neural/iou_net_jitter.pth")
    scores = est.score_frame(image_tensor, boxes_xyxy)  # → dict
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROI_DIM     = 12544          # 256 × 7 × 7  (same for all MobileNetV3-FPN variants)
ROI_OUTPUT  = (7, 7)
CONF_THR    = 0.5            # UA-DETRAC / UAVDT default (matches training in train_iounet.py)
CONF_THR_MOT17 = 0.1        # MOT17
IOF_THR     = 0.5
MAX_BOXES   = 100
VEHICLE_IDS = {3, 6, 8}      # COCO: car=3, bus=6, truck=8  (motorcycle=4 optional)
PERSON_IDS  = {1}             # COCO: person (MOT17)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class IoUPredictor(nn.Module):
    def __init__(self, in_dim: int = ROI_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),   nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# RoI helpers
# ---------------------------------------------------------------------------
def _roi_features(det_model, imgs_t: torch.Tensor,
                  boxes_list: List[torch.Tensor],
                  img_shapes: List[Tuple[int, int]],
                  cached_features=None) -> torch.Tensor:
    """Extract 7×7 RoI-pooled features from FasterRCNN backbone.

    If cached_features is provided (captured via a backbone forward hook
    during the detector's own forward pass), the backbone is NOT re-run.
    This is the intended usage in benchmarks to avoid double backbone cost.
    """
    with torch.no_grad():
        features = cached_features if cached_features is not None \
                   else det_model.backbone(imgs_t)

    # Build [img_idx, x1, y1, x2, y2] tensor
    rois = []
    for i, boxes in enumerate(boxes_list):
        if len(boxes) == 0:
            continue
        idx = torch.full((len(boxes), 1), float(i), device=boxes.device)
        rois.append(torch.cat([idx, boxes], dim=1))

    if not rois:
        return torch.zeros(0, ROI_DIM, device=imgs_t.device)

    rois_t = torch.cat(rois, dim=0)

    # Scale rois to feature-map space
    h_img, w_img = img_shapes[0]
    pooled = det_model.roi_heads.box_roi_pool(
        features, [rois_t[:, 1:]], [(h_img, w_img)])
    return pooled.flatten(start_dim=1)


def _iof_xyxy(pred: List[float], ign: List[float]) -> float:
    """Intersection over pred-area (ignored-region filter)."""
    px1, py1, px2, py2 = pred
    ix1, iy1, ix2, iy2 = ign
    inter = max(0.0, min(px2, ix2) - max(px1, ix1)) * \
            max(0.0, min(py2, iy2) - max(py1, iy1))
    area  = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    return inter / area if area > 0 else 0.0


def _filter_boxes(output: dict, ignored_xyxy: torch.Tensor,
                  conf_thr: float, max_n: int, class_ids: set = None):
    if class_ids is None:
        class_ids = VEHICLE_IDS
    boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
    if boxes.numel() == 0:
        return boxes, scores

    keep = (scores >= conf_thr) & torch.tensor(
        [int(l.item()) in class_ids for l in labels],
        dtype=torch.bool, device=scores.device)
    boxes, scores = boxes[keep], scores[keep]

    if ignored_xyxy is not None and ignored_xyxy.numel() > 0:
        valid = [
            max((_iof_xyxy(b.cpu().tolist(), ig.cpu().tolist())
                 for ig in ignored_xyxy), default=0.0) <= IOF_THR
            for b in boxes
        ]
        mask  = torch.tensor(valid, dtype=torch.bool, device=boxes.device)
        boxes, scores = boxes[mask], scores[mask]

    if len(boxes) > max_n:
        boxes, scores = boxes[:max_n], scores[:max_n]
    return boxes, scores


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------
class IoUNetEstimator:
    """
    Wraps IoUPredictor for per-frame quality scoring.

    Parameters
    ----------
    weights_path  : path to .pth checkpoint (fc1/fc2/fc3 key format)
    det_model     : a loaded FasterRCNN detector (torchvision); if None, loads
                    fasterrcnn_mobilenet_v3_large_fpn or _320_fpn based on use_320
    device        : 'cuda' | 'cpu'
    conf_thr      : confidence threshold for box filtering (0.5 for UA-DETRAC/UAVDT,
                    0.1 for MOT17)
    class_ids     : set of COCO class IDs to keep (VEHICLE_IDS for vehicles,
                    PERSON_IDS for MOT17)
    use_320       : if True, load fasterrcnn_mobilenet_v3_large_320_fpn backbone
    """

    _KEY_MAP = {'fc1': 'net.0', 'fc2': 'net.2', 'fc3': 'net.4'}

    def __init__(self, weights_path: str,
                 det_model=None, device: str = 'cuda',
                 conf_thr: float = CONF_THR,
                 class_ids: set = None,
                 use_320: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thr  = conf_thr
        self.class_ids = class_ids if class_ids is not None else VEHICLE_IDS

        # Load detector
        if det_model is None:
            if use_320:
                from torchvision.models.detection import (
                    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                    fasterrcnn_mobilenet_v3_large_320_fpn,
                )
                w = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.det_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).to(self.device)
            else:
                from torchvision.models.detection import (
                    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                    fasterrcnn_mobilenet_v3_large_fpn,
                )
                w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                self.det_model = fasterrcnn_mobilenet_v3_large_fpn(weights=w).to(self.device)
            self.transform  = w.transforms()
        else:
            self.det_model = det_model.to(self.device)
            self.transform  = None
        self.det_model.eval()

        # Load quality head
        ckpt = torch.load(weights_path, map_location=self.device)
        mapped = {}
        for k, v in ckpt.items():
            prefix  = k.split('.')[0]
            new_key = self._KEY_MAP.get(prefix, prefix) + '.' + '.'.join(k.split('.')[1:])
            mapped[new_key] = v
        self.iou_head = IoUPredictor(ROI_DIM).to(self.device)
        self.iou_head.load_state_dict(mapped)
        self.iou_head.eval()

    @torch.no_grad()
    def score_frame(self, img_tensor: torch.Tensor,
                    ignored_xyxy: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Score one frame.

        Parameters
        ----------
        img_tensor   : CHW float tensor (already transformed) on any device
        ignored_xyxy : (N,4) tensor of ignored-region boxes in xyxy format, or None

        Returns
        -------
        dict with keys: iou_net_mean, iou_net_max, iou_net_top3
        """
        imgs_t = img_tensor.unsqueeze(0).to(self.device)
        if ignored_xyxy is None:
            ignored_xyxy = torch.zeros(0, 4, device=self.device)
        else:
            ignored_xyxy = ignored_xyxy.to(self.device)

        # Capture backbone features and transformed image size via hooks.
        # FasterRCNN internally normalizes and resizes before backbone, so we
        # must use those features (not a raw second backbone pass) for correct ROI pooling.
        _backbone_out = {}
        _trans_size   = {}

        def _feat_hook(module, inp, out):
            _backbone_out['features'] = out

        def _transform_hook(module, inp, out):
            _trans_size['sizes'] = out[0].image_sizes  # [(H_t, W_t), ...]

        h1 = self.det_model.backbone.register_forward_hook(_feat_hook)
        h2 = self.det_model.transform.register_forward_hook(_transform_hook)
        try:
            out = self.det_model(imgs_t)[0]
        finally:
            h1.remove()
            h2.remove()

        orig_h, orig_w = imgs_t.shape[-2:]
        boxes, scores = _filter_boxes(out, ignored_xyxy, self.conf_thr, MAX_BOXES,
                                      class_ids=self.class_ids)

        if len(boxes) == 0:
            return {'iou_net_mean': 0.0, 'iou_net_max': 0.0, 'iou_net_top3': 0.0,
                    'iou_net_conf_weighted_mean': 0.0, 'num_boxes': 0}

        # Scale boxes from original image space → transformed (backbone) space.
        trans_h, trans_w = _trans_size['sizes'][0]
        sx = trans_w / orig_w; sy = trans_h / orig_h
        scale = torch.tensor([sx, sy, sx, sy], device=boxes.device)
        boxes_trans = boxes * scale

        features = _backbone_out['features']
        roi  = _roi_features(self.det_model, imgs_t, [boxes_trans], [(trans_h, trans_w)],
                             cached_features=features)
        pred = torch.clamp(self.iou_head(roi), 0.0, 1.0).cpu().numpy()
        conf = scores.cpu().numpy()

        top3 = np.sort(pred)[::-1][:min(3, len(pred))]
        conf_sum = float(np.sum(conf))
        conf_weighted = float(np.sum(pred * conf) / conf_sum) if conf_sum > 1e-12 else float(np.mean(pred))
        return {
            'iou_net_mean': float(np.mean(pred)),
            'iou_net_max':  float(np.max(pred)),
            'iou_net_top3': float(np.mean(top3)),
            'iou_net_conf_weighted_mean': conf_weighted,
            'num_boxes':    len(pred),
        }
