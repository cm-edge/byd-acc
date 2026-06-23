#!/usr/bin/env python3
"""
Train IoU-Net quality predictor on FasterRCNN-MobileNetV3 for UA-DETRAC.

IoU-Net learns to predict max IoU between each detected box and GT boxes,
using RoI-pooled features from the detector backbone. The per-frame aggregated
score (mean/max/top3 predicted IoU) serves as a quality indicator:
  high score = weak model is already doing well on this frame
  low  score = frame is hard for the weak model → candidate for offloading

Train set: UA-DETRAC official train split
Test  set: UA-DETRAC official test split + zero-shot on UAVDT/MOT17
           (run scoring/compute_iounet_gfl.py for cross-dataset inference)

Output:
  weights/neural/iou_net.pth          — trained checkpoint
  scores/iounet/ua-detrac/{video}.csv — per-video quality scores

Usage:
    python training/train_iounet.py
    python training/train_iounet.py --epochs 5 --batch-size 4
"""

import argparse
import os
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
# Paths
# ---------------------------------------------------------------------------
# UA-DETRAC dataset (download into <repo>/data/; see detection/README.md).
IMAGES_DIR  = PROJECT_ROOT / "data" / "UA-DETRAC" / "DETRAC-Images"
ANNO_BASE   = PROJECT_ROOT / "data" / "UA-DETRAC" / "DETRAC-Annos"
TRAIN_ANNO  = ANNO_BASE / "DETRAC-Train-Annotations-XML"
TEST_ANNO   = ANNO_BASE / "DETRAC-Test-Annotations-XML"

MODEL_NAME  = "FASTERRCNN-MOBILENET-V3-LARGE-FPN"
CONF_LABEL  = "conf_0_1"

VEHICLE_CLASSES = {3, 4, 6, 8}   # COCO: car, motorcycle, bus, truck
CONF_THR        = 0.5
IOF_THR         = 0.5
MAX_BOXES       = 100
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# IoU-Net model
# ---------------------------------------------------------------------------
class IoUPredictor(nn.Module):
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
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou_xyxy(b1, b2) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    a1 = max(0.0, float(b1[2])-float(b1[0])) * max(0.0, float(b1[3])-float(b1[1]))
    a2 = max(0.0, float(b2[2])-float(b2[0])) * max(0.0, float(b2[3])-float(b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _iof_xyxy(pred, ign) -> float:
    px1,py1,px2,py2 = [float(v) for v in pred]
    ix1,iy1,ix2,iy2 = [float(v) for v in ign]
    inter = max(0.0, min(px2,ix2)-max(px1,ix1)) * max(0.0, min(py2,iy2)-max(py1,iy1))
    area  = max(0.0, px2-px1) * max(0.0, py2-py1)
    return inter / area if area > 0 else 0.0


def _max_iou_to_gt(box, gt_boxes: torch.Tensor) -> float:
    if gt_boxes is None or gt_boxes.numel() == 0:
        return 0.0
    pb = box.detach().cpu().numpy().tolist()
    return max(_iou_xyxy(pb, gt.tolist()) for gt in gt_boxes.cpu())


# ---------------------------------------------------------------------------
# GT XML parsing
# ---------------------------------------------------------------------------
def _find_anno(video: str) -> Optional[Path]:
    for d in [TRAIN_ANNO, TEST_ANNO]:
        p = d / f"{video.upper()}.xml"
        if p.exists():
            return p
    return None


def parse_gt_xml(xml_path: Path) -> Dict[int, Dict]:
    """Returns frame_idx -> {gt_boxes: [...xyxy], ignored_boxes: [...xyxy]}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ignored: List[BBox] = []
    ign_node = root.find("ignored_region")
    if ign_node is not None:
        for b in ign_node.findall("box"):
            l  = float(b.get("left",   "0"))
            t  = float(b.get("top",    "0"))
            w  = float(b.get("width",  "0"))
            h  = float(b.get("height", "0"))
            if w > 0 and h > 0:
                ignored.append((l, t, l+w, t+h))

    frames = {}
    for frame in root.findall("frame"):
        fid = int(frame.get("num"))
        gt  = []
        tl  = frame.find("target_list")
        if tl is not None:
            for target in tl.findall("target"):
                b = target.find("box")
                if b is None:
                    continue
                l  = float(b.get("left",   "0"))
                t  = float(b.get("top",    "0"))
                w  = float(b.get("width",  "0"))
                h  = float(b.get("height", "0"))
                if w > 0 and h > 0:
                    gt.append((l, t, l+w, t+h))
        frames[fid] = {"gt_boxes": gt, "ignored_boxes": ignored}
    return frames


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DETRACDataset(torch.utils.data.Dataset):
    def __init__(self, video_list: List[str], transform):
        self.transform = transform
        self.samples   = []
        self.anno: Dict[str, Dict[int, Dict]] = {}

        for video in tqdm(video_list, desc="Loading GT"):
            anno_path = _find_anno(video)
            if anno_path is None:
                continue
            img_dir = IMAGES_DIR / video.upper()
            if not img_dir.exists():
                continue
            self.anno[video] = parse_gt_xml(anno_path)

            for img_file in sorted(img_dir.glob("*.jpg")):
                try:
                    nums = [c for c in img_file.stem if c.isdigit()]
                    fid  = int(''.join(nums))
                except ValueError:
                    continue
                self.samples.append({
                    "img_path": img_file,
                    "video": video,
                    "frame_id": fid,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        ow, oh = img.size
        t   = self.transform(img)
        nh, nw = t.shape[-2:]

        sx, sy = nw / float(ow), nh / float(oh)

        frame_gt = self.anno.get(s["video"], {}).get(
            s["frame_id"], {"gt_boxes": [], "ignored_boxes": []})

        def scale(boxes):
            return [(x1*sx, y1*sy, x2*sx, y2*sy) for x1,y1,x2,y2 in boxes]

        gt_t  = torch.tensor(scale(frame_gt["gt_boxes"]),      dtype=torch.float32)
        ign_t = torch.tensor(scale(frame_gt["ignored_boxes"]), dtype=torch.float32)

        return t, {"gt": gt_t, "ignored": ign_t}, s


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------
def _filter_boxes(output, ignored: torch.Tensor, conf_thr: float, max_n: int):
    boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
    if boxes.numel() == 0:
        return boxes, scores

    keep = (scores >= conf_thr) & torch.tensor(
        [int(l.item()) in VEHICLE_CLASSES for l in labels],
        dtype=torch.bool, device=scores.device)
    boxes, scores = boxes[keep], scores[keep]

    # Filter by ignored region (IOF > IOF_THR)
    if ignored is not None and ignored.numel() > 0:
        valid = []
        for b in boxes:
            max_iof = max((_iof_xyxy(b.cpu().tolist(), ig.cpu().tolist())
                           for ig in ignored), default=0.0)
            valid.append(max_iof <= IOF_THR)
        mask = torch.tensor(valid, dtype=torch.bool, device=boxes.device)
        boxes, scores = boxes[mask], scores[mask]

    if len(boxes) > max_n:
        boxes, scores = boxes[:max_n], scores[:max_n]
    return boxes, scores


def _roi_features(det_model, imgs_t: torch.Tensor,
                  boxes_list: List[torch.Tensor],
                  img_shapes: List[Tuple[int, int]]) -> torch.Tensor:
    feats = det_model.backbone(imgs_t)
    return det_model.roi_heads.box_roi_pool(feats, boxes_list, img_shapes)


def _infer_roi_dim(det_model) -> int:
    dummy = torch.zeros(1, 3, 320, 320, device=DEVICE)
    with torch.no_grad():
        feats = det_model.backbone(dummy)
        roi   = det_model.roi_heads.box_roi_pool(
            feats, [torch.tensor([[0,0,10,10]], dtype=torch.float32, device=DEVICE)], [(320,320)])
    return int(roi.flatten(start_dim=1).shape[1])


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(iou_model, det_model, loader, optimizer, epoch: int):
    iou_model.train()
    losses, n_boxes = [], 0

    for imgs, targets, _ in tqdm(loader, desc=f"Train epoch {epoch}"):
        imgs_t  = torch.stack([i.to(DEVICE) for i in imgs])
        shapes  = [tuple(i.shape[-2:]) for i in imgs]

        with torch.no_grad():
            outs = det_model(imgs_t)

        boxes_list, labels_list = [], []
        for i, out in enumerate(outs):
            ign = targets[i]["ignored"].to(DEVICE)
            gt  = targets[i]["gt"].to(DEVICE)
            b, s = _filter_boxes(out, ign, CONF_THR, MAX_BOXES)
            boxes_list.append(b.detach())
            if len(b):
                labels = torch.tensor(
                    [_max_iou_to_gt(b[k], gt) for k in range(len(b))],
                    dtype=torch.float32, device=DEVICE)
                labels_list.append(labels)

        if not any(len(b) > 0 for b in boxes_list):
            continue

        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        y   = torch.cat([l for b, l in zip(boxes_list, labels_list) if len(b)], dim=0)

        if roi.shape[0] != y.shape[0]:
            # Skip malformed batch
            continue

        loss = F.smooth_l1_loss(iou_model(roi), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        n_boxes += int(y.numel())

    return np.mean(losses) if losses else float('nan'), n_boxes


# ---------------------------------------------------------------------------
# Inference → per-frame aggregated scores
# ---------------------------------------------------------------------------
@torch.no_grad()
def inference_frame_scores(iou_model, det_model, loader) -> pd.DataFrame:
    iou_model.eval()
    rows = []

    for imgs, targets, metas in tqdm(loader, desc="Inference"):
        imgs_t  = torch.stack([i.to(DEVICE) for i in imgs])
        shapes  = [tuple(i.shape[-2:]) for i in imgs]
        outs    = det_model(imgs_t)

        boxes_list = []
        meta_boxes = []

        for i, out in enumerate(outs):
            ign = targets[i]["ignored"].to(DEVICE)
            b, s = _filter_boxes(out, ign, CONF_THR, MAX_BOXES)
            boxes_list.append(b.detach())
            if len(b):
                for k in range(len(b)):
                    meta_boxes.append({
                        "video":    metas[i]["video"],
                        "frame_id": metas[i]["frame_id"],
                        "conf":     float(s[k].cpu()),
                    })

        if not meta_boxes:
            # No detections in batch → add zero-score rows for each frame
            for m in metas:
                rows.append({
                    "video": m["video"], "frame_id": m["frame_id"],
                    "iou_net_mean": 0.0, "iou_net_max": 0.0, "iou_net_top3": 0.0,
                })
            continue

        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        pred_iou = torch.clamp(iou_model(roi), 0.0, 1.0).cpu().numpy()

        assert len(pred_iou) == len(meta_boxes), \
            f"Pred/meta mismatch: {len(pred_iou)} vs {len(meta_boxes)}"

        # Attach predictions to meta
        for k, m in enumerate(meta_boxes):
            m["pred_iou"] = float(pred_iou[k])

        # Aggregate to frame level
        box_df = pd.DataFrame(meta_boxes)
        for (video, fid), g in box_df.groupby(["video", "frame_id"]):
            piou = g["pred_iou"].values
            top3 = np.sort(piou)[::-1][:min(3, len(piou))]
            rows.append({
                "video":        video,
                "frame_id":     int(fid),
                "iou_net_mean": float(np.mean(piou)),
                "iou_net_max":  float(np.max(piou)),
                "iou_net_top3": float(np.mean(top3)),
            })

        # Frames with no detections
        detected_fids = set(box_df["frame_id"].values)
        for m in metas:
            if m["frame_id"] not in detected_fids:
                rows.append({
                    "video": m["video"], "frame_id": m["frame_id"],
                    "iou_net_mean": 0.0, "iou_net_max": 0.0, "iou_net_top3": 0.0,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train IoU-Net quality predictor')
    parser.add_argument('--epochs',    type=int,   default=5)
    parser.add_argument('--batch-size',type=int,   default=4)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--conf-label',type=str,   default=CONF_LABEL)
    parser.add_argument('--output-dir',type=str,   default=None)
    parser.add_argument('--weights',   type=str,   default=None,
                        help='Path to pre-trained .pth checkpoint; skips training '
                             '(supports fc1/fc2/fc3 or net.0/net.2/net.4 key names)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
                 PROJECT_ROOT / "scores" / "iounet" / "ua-detrac"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device:     {DEVICE}")
    print(f"Output dir: {output_dir}")

    # Load FasterRCNN-MobileNetV3
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    weights  = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    det_model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(DEVICE)
    det_model.eval()
    transform = weights.transforms()

    if not hasattr(det_model, 'roi_heads') or \
       not hasattr(det_model.roi_heads, 'box_roi_pool'):
        raise RuntimeError("Detector has no roi_heads.box_roi_pool — need Faster R-CNN.")

    # UA-DETRAC split
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

    # Build IoU-Net
    roi_dim   = 12544  # 256 × 7 × 7, same for all MobileNetV3-FPN variants
    print(f"RoI feature dim: {roi_dim}")
    iou_model = IoUPredictor(roi_dim).to(DEVICE)

    if args.weights:
        # Load pre-trained checkpoint, remapping fc1/fc2/fc3 → net.0/net.2/net.4
        ckpt = torch.load(args.weights, map_location=DEVICE)
        key_map = {'fc1': 'net.0', 'fc2': 'net.2', 'fc3': 'net.4'}
        mapped = {}
        for k, v in ckpt.items():
            prefix = k.split('.')[0]
            new_k = key_map.get(prefix, prefix) + '.' + '.'.join(k.split('.')[1:])
            mapped[new_k] = v
        iou_model.load_state_dict(mapped)
        print(f"Loaded weights from: {args.weights}")
    else:
        optimizer = torch.optim.Adam(iou_model.parameters(), lr=args.lr)
        print("\n=== Training IoU-Net ===")
        for ep in range(1, args.epochs + 1):
            avg_loss, n_boxes = train_one_epoch(iou_model, det_model, train_loader, optimizer, ep)
            print(f"  Epoch {ep}/{args.epochs}  loss={avg_loss:.5f}  boxes={n_boxes}")
        weights_dir = PROJECT_ROOT / "weights" / "neural"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / "iou_net.pth"
        torch.save(iou_model.state_dict(), model_path)
        print(f"Saved model: {model_path}")

    # Inference on test split → save per-video CSVs
    print("\n=== Inference on test split ===")
    frame_df = inference_frame_scores(iou_model, det_model, test_loader)

    for video, g in frame_df.groupby("video"):
        out_path = output_dir / f"{video}.csv"
        g[["frame_id","iou_net_mean","iou_net_max","iou_net_top3"]].to_csv(
            out_path, index=False, float_format='%.6f')
        print(f"  {video}: {len(g)} frames  mean={g['iou_net_mean'].mean():.4f}")

    print(f"\nDone. Scores saved to: {output_dir}")


if __name__ == '__main__':
    main()
