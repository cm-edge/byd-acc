#!/usr/bin/env python3
"""
Train GFL (Generalized Focal Loss) quality predictor on FasterRCNN-MobileNetV3.

GFL learns a quality-aware score using Quality Focal Loss:
  - matched box  (IoU >= threshold) → target = IoU(pred, GT)
  - unmatched box                   → target = 0
The frame-level mean quality score serves as a quality indicator.

Output: per-video CSV with columns [frame_id, gfl_mean, gfl_max, gfl_top3].

Usage:
    python training/train_gfl.py
    python training/train_gfl.py --epochs 5 --batch-size 4
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT / "estimator"))
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))

from gt_parser import get_video_split

# ---------------------------------------------------------------------------
# Re-use data infrastructure from train_iounet_predictor
# ---------------------------------------------------------------------------
from train_iounet import (
    IMAGES_DIR, TRAIN_ANNO, TEST_ANNO, VEHICLE_CLASSES,
    CONF_THR, IOF_THR, MAX_BOXES, DEVICE, CONF_LABEL,
    BBox, DETRACDataset, _collate,
    _filter_boxes, _roi_features, _infer_roi_dim, _max_iou_to_gt,
)

MODEL_NAME = "FASTERRCNN-MOBILENET-V3-LARGE-FPN"

MATCH_IOU_THR = 0.5
QFL_BETA      = 2.0


# ---------------------------------------------------------------------------
# GFL model
# ---------------------------------------------------------------------------
class GFLPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),   nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# Quality Focal Loss
# ---------------------------------------------------------------------------
def quality_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       beta: float = QFL_BETA) -> torch.Tensor:
    """L_QFL = |y - sigmoid(logit)|^beta * BCE(logit, y)"""
    p   = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (torch.abs(targets - p) ** beta * bce).mean()


def _make_gfl_label(box, gt_boxes: torch.Tensor, thr: float = MATCH_IOU_THR) -> float:
    iou = _max_iou_to_gt(box, gt_boxes)
    return float(iou) if iou >= thr else 0.0


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(gfl_model, det_model, loader, optimizer, epoch: int):
    gfl_model.train()
    losses = []

    for imgs, targets, _ in tqdm(loader, desc=f"Train epoch {epoch}"):
        imgs_t = torch.stack([i.to(DEVICE) for i in imgs])
        shapes = [tuple(i.shape[-2:]) for i in imgs]

        with torch.no_grad():
            outs = det_model(imgs_t)

        boxes_list, labels_list = [], []
        for i, out in enumerate(outs):
            ign = targets[i]["ignored"].to(DEVICE)
            gt  = targets[i]["gt"].to(DEVICE)
            b, _ = _filter_boxes(out, ign, CONF_THR, MAX_BOXES)
            boxes_list.append(b.detach())
            if len(b):
                labels = torch.tensor(
                    [_make_gfl_label(b[k], gt) for k in range(len(b))],
                    dtype=torch.float32, device=DEVICE)
                labels_list.append(labels)

        if not any(len(b) > 0 for b in boxes_list):
            continue

        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        y   = torch.cat([l for b, l in zip(boxes_list, labels_list) if len(b)], dim=0)

        if roi.shape[0] != y.shape[0]:
            continue

        loss = quality_focal_loss(gfl_model(roi), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return np.mean(losses) if losses else float('nan')


# ---------------------------------------------------------------------------
# Inference → per-frame scores
# ---------------------------------------------------------------------------
@torch.no_grad()
def inference_frame_scores(gfl_model, det_model, loader) -> pd.DataFrame:
    gfl_model.eval()
    rows = []

    for imgs, targets, metas in tqdm(loader, desc="Inference"):
        imgs_t = torch.stack([i.to(DEVICE) for i in imgs])
        shapes = [tuple(i.shape[-2:]) for i in imgs]
        outs   = det_model(imgs_t)

        boxes_list, meta_boxes = [], []
        for i, out in enumerate(outs):
            ign = targets[i]["ignored"].to(DEVICE)
            b, s = _filter_boxes(out, ign, CONF_THR, MAX_BOXES)
            boxes_list.append(b.detach())
            if len(b):
                for k in range(len(b)):
                    meta_boxes.append({
                        "video": metas[i]["video"],
                        "frame_id": metas[i]["frame_id"],
                    })

        if not meta_boxes:
            for m in metas:
                rows.append({"video": m["video"], "frame_id": m["frame_id"],
                             "gfl_mean": 0.0, "gfl_max": 0.0, "gfl_top3": 0.0})
            continue

        roi  = _roi_features(det_model, imgs_t, boxes_list, shapes)
        pred = torch.sigmoid(gfl_model(roi)).cpu().numpy()

        assert len(pred) == len(meta_boxes)
        for k, m in enumerate(meta_boxes):
            m["gfl_quality"] = float(pred[k])

        box_df = pd.DataFrame(meta_boxes)
        for (video, fid), g in box_df.groupby(["video", "frame_id"]):
            q    = g["gfl_quality"].values
            top3 = np.sort(q)[::-1][:min(3, len(q))]
            rows.append({
                "video":    video,
                "frame_id": int(fid),
                "gfl_mean": float(np.mean(q)),
                "gfl_max":  float(np.max(q)),
                "gfl_top3": float(np.mean(top3)),
            })

        detected_fids = set(box_df["frame_id"].values)
        for m in metas:
            if m["frame_id"] not in detected_fids:
                rows.append({"video": m["video"], "frame_id": m["frame_id"],
                             "gfl_mean": 0.0, "gfl_max": 0.0, "gfl_top3": 0.0})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train GFL quality predictor')
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--conf-label', type=str,   default=CONF_LABEL)
    parser.add_argument('--output-dir', type=str,   default=None)
    parser.add_argument('--weights',    type=str,   default=None,
                        help='Path to pre-trained .pth checkpoint; skips training '
                             '(supports fc1/fc2/fc3 or net.0/net.2/net.4 key names)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
                 PROJECT_ROOT / "scores" / "gfl" / "ua-detrac"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device:     {DEVICE}")
    print(f"Output dir: {output_dir}")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    weights   = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    det_model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(DEVICE)
    det_model.eval()
    transform = weights.transforms()

    all_videos = [d.name for d in sorted(IMAGES_DIR.iterdir()) if d.is_dir()]
    train_vids = [v for v in all_videos if get_video_split(v) == 'train']
    test_vids  = [v for v in all_videos if get_video_split(v) == 'test']
    print(f"Train: {len(train_vids)} videos,  Test: {len(test_vids)} videos")

    train_ds = DETRACDataset(train_vids, transform)
    test_ds  = DETRACDataset(test_vids,  transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate)

    roi_dim   = 12544  # 256 × 7 × 7, same for all MobileNetV3-FPN variants
    print(f"RoI feature dim: {roi_dim}")
    gfl_model = GFLPredictor(roi_dim).to(DEVICE)

    if args.weights:
        ckpt = torch.load(args.weights, map_location=DEVICE)
        key_map = {'fc1': 'net.0', 'fc2': 'net.2', 'fc3': 'net.4'}
        mapped = {}
        for k, v in ckpt.items():
            prefix = k.split('.')[0]
            new_k = key_map.get(prefix, prefix) + '.' + '.'.join(k.split('.')[1:])
            mapped[new_k] = v
        gfl_model.load_state_dict(mapped)
        print(f"Loaded weights from: {args.weights}")
    else:
        optimizer = torch.optim.Adam(gfl_model.parameters(), lr=args.lr)
        print("\n=== Training GFL predictor ===")
        for ep in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(gfl_model, det_model, train_loader, optimizer, ep)
            print(f"  Epoch {ep}/{args.epochs}  loss={avg_loss:.5f}")
        weights_dir = PROJECT_ROOT / "weights" / "neural"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / "gfl.pth"
        torch.save(gfl_model.state_dict(), model_path)
        print(f"Saved model: {model_path}")

    print("\n=== Inference on test split ===")
    frame_df = inference_frame_scores(gfl_model, det_model, test_loader)

    for video, g in frame_df.groupby("video"):
        out_path = output_dir / f"{video}.csv"
        g[["frame_id","gfl_mean","gfl_max","gfl_top3"]].to_csv(
            out_path, index=False, float_format='%.6f')
        print(f"  {video}: {len(g)} frames  mean={g['gfl_mean'].mean():.4f}")

    print(f"\nDone. Scores saved to: {output_dir}")


if __name__ == '__main__':
    main()
