"""
GFL (Quality Focal Loss) quality estimator — inference only.

Same backbone and RoI-pooling as IoUNet, but trained with Quality Focal Loss:
  matched box (IoU ≥ 0.5 with GT) → target = IoU value
  unmatched box                    → target = 0
Output is sigmoid(logit) ∈ (0, 1).

High gfl_mean  →  weak model quality high  →  no need to offload.
Low  gfl_mean  →  weak model unreliable    →  offload to strong model.

Usage:
    from gfl_inference import GFLEstimator
    est = GFLEstimator("weights/neural/gfl_qfl_quality_head.pth")
    scores = est.score_frame(image_tensor)   # → dict
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Re-use shared helpers from iounet_inference
from iounet_inference import (
    ROI_DIM, ROI_OUTPUT, CONF_THR, IOF_THR, MAX_BOXES, VEHICLE_IDS,
    _filter_boxes, _roi_features,
)

# ---------------------------------------------------------------------------
# Model (identical architecture to IoUPredictor; outputs logits → sigmoid)
# ---------------------------------------------------------------------------
class GFLPredictor(nn.Module):
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
# Main estimator class
# ---------------------------------------------------------------------------
class GFLEstimator:
    """
    Wraps GFLPredictor for per-frame quality scoring.

    Parameters
    ----------
    weights_path  : path to .pth checkpoint (fc1/fc2/fc3 key format)
    det_model     : a loaded FasterRCNN detector; if None, loads automatically
    device        : 'cuda' | 'cpu'
    conf_thr      : confidence threshold (0.5 UA-DETRAC/UAVDT, 0.1 MOT17)
    class_ids     : COCO class IDs to keep (VEHICLE_IDS or PERSON_IDS)
    use_320       : if True, load fasterrcnn_mobilenet_v3_large_320_fpn
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

        ckpt = torch.load(weights_path, map_location=self.device)
        mapped = {}
        for k, v in ckpt.items():
            prefix  = k.split('.')[0]
            new_key = self._KEY_MAP.get(prefix, prefix) + '.' + '.'.join(k.split('.')[1:])
            mapped[new_key] = v
        self.gfl_head = GFLPredictor(ROI_DIM).to(self.device)
        self.gfl_head.load_state_dict(mapped)
        self.gfl_head.eval()

    @torch.no_grad()
    def score_frame(self, img_tensor: torch.Tensor,
                    ignored_xyxy: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Score one frame.

        Parameters
        ----------
        img_tensor   : CHW float tensor (already transformed)
        ignored_xyxy : (N,4) xyxy ignored-region boxes, or None

        Returns
        -------
        dict with keys: gfl_mean, gfl_max, gfl_top3, num_boxes
        """
        imgs_t = img_tensor.unsqueeze(0).to(self.device)
        if ignored_xyxy is None:
            ignored_xyxy = torch.zeros(0, 4, device=self.device)
        else:
            ignored_xyxy = ignored_xyxy.to(self.device)

        # Capture backbone features and transformed size via hooks (same fix as IoUNet).
        _backbone_out = {}
        _trans_size   = {}

        def _feat_hook(module, inp, out):
            _backbone_out['features'] = out

        def _transform_hook(module, inp, out):
            _trans_size['sizes'] = out[0].image_sizes

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
            return {'gfl_mean': 0.0, 'gfl_max': 0.0, 'gfl_top3': 0.0,
                    'gfl_final_score_mean': 0.0, 'num_boxes': 0}

        trans_h, trans_w = _trans_size['sizes'][0]
        sx = trans_w / orig_w; sy = trans_h / orig_h
        scale = torch.tensor([sx, sy, sx, sy], device=boxes.device)
        boxes_trans = boxes * scale

        roi  = _roi_features(self.det_model, imgs_t, [boxes_trans], [(trans_h, trans_w)],
                             cached_features=_backbone_out['features'])
        pred = torch.sigmoid(self.gfl_head(roi)).cpu().numpy()
        conf = scores.cpu().numpy()

        top3 = np.sort(pred)[::-1][:min(3, len(pred))]
        final_score = pred * conf          # element-wise: quality × confidence
        return {
            'gfl_mean':              float(np.mean(pred)),
            'gfl_max':               float(np.max(pred)),
            'gfl_top3':              float(np.mean(top3)),
            'gfl_final_score_mean':  float(np.mean(final_score)),
            'num_boxes':             len(pred),
        }
