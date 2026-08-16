#!/usr/bin/env python3
"""
Generalized GFL/IoUNet baseline training and inference for multiple datasets.

Trains GFL (Quality Focal Loss) or IoUNet (predicted IoU regression) on a
given dataset's train split and runs inference on the test split. Saves
per-frame scores in the same format as train_gfl_predictor.py / train_iounet_predictor.py.

Supported datasets: uavdt, imagenet-vid
For MOT17: pass --weights with a pre-trained .pth and --inference-only.

Usage:
    # Train on UAVDT
    python training/train_baselines_multidataset.py --dataset uavdt --method gfl
    python training/train_baselines_multidataset.py --dataset uavdt --method iounet

    # MOT17 inference-only with pre-trained weights
    python training/train_baselines_multidataset.py --dataset mot17 --method gfl \\
        --weights /path/to/gfl.pth --inference-only
"""

import argparse
import json
import sys
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
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "dataset"))

from adapters import get_adapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THR  = 0.1
IOF_THR   = 0.5
MAX_BOXES = 100
MATCH_IOU_THR = 0.5
QFL_BETA  = 2.0

MODEL_WEAK = "FASTERRCNN-MOBILENET-V3-LARGE-FPN"
CONF_LABEL = "conf_0_1"

# COCO vehicle class IDs (car=3, motorcycle=4, bus=6, truck=8)
VEHICLE_COCO = {3, 4, 6, 8}
# MOT17 person class
PERSON_COCO = {1}

DATA_ROOTS = {
    "ua-detrac":    PROJECT_ROOT / "data" / "UA-DETRAC",
    "uavdt":        PROJECT_ROOT / "data" / "UAVDT",
    "mot17":        PROJECT_ROOT / "data" / "MOT17",
    "imagenet-vid": PROJECT_ROOT / "data" / "ILSVRC",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou_ltwh(b1, b2) -> float:
    l1, t1, w1, h1 = b1
    l2, t2, w2, h2 = b2
    ix1, iy1 = max(l1, l2), max(t1, t2)
    ix2, iy2 = min(l1+w1, l2+w2), min(t1+h1, t2+h2)
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.


def _iof_ltwh(pred_ltwh, ign_ltwh) -> float:
    pl, pt, pw, ph = pred_ltwh
    il, it, iw, ih = ign_ltwh
    ix1, iy1 = max(pl, il), max(pt, it)
    ix2, iy2 = min(pl+pw, il+iw), min(pt+ph, it+ih)
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    area = pw * ph
    return inter / area if area > 0 else 0.


def _max_iou_to_gt(box_xyxy, gt_ltwh_list) -> float:
    if not gt_ltwh_list:
        return 0.
    bx1, by1, bx2, by2 = box_xyxy
    bltwh = (bx1, by1, bx2-bx1, by2-by1)
    return max(_iou_ltwh(bltwh, g) for g in gt_ltwh_list)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GFLPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten(start_dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)


class IoUPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten(start_dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)


def _quality_focal_loss(logits, targets, beta=QFL_BETA):
    p   = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return ((targets - p).abs() ** beta * bce).mean()


# ---------------------------------------------------------------------------
# Detector helpers
# ---------------------------------------------------------------------------
def _load_detector():
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model   = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(DEVICE)
    model.eval()
    return model, weights.transforms()


def _infer_roi_dim(det_model) -> int:
    dummy = torch.zeros(1, 3, 320, 320, device=DEVICE)
    with torch.no_grad():
        feats = det_model.backbone(dummy)
        roi   = det_model.roi_heads.box_roi_pool(
            feats,
            [torch.tensor([[0,0,10,10]], dtype=torch.float32, device=DEVICE)],
            [(320, 320)],
        )
    return int(roi.flatten(start_dim=1).shape[1])


def _roi_features(det_model, imgs_t, boxes_list, img_shapes):
    feats = det_model.backbone(imgs_t)
    return det_model.roi_heads.box_roi_pool(feats, boxes_list, img_shapes)


def _filter_detections(output, ignored_ltwh, conf_thr, max_n, vehicle_class_ids):
    boxes  = output["boxes"]
    scores = output["scores"]
    labels = output["labels"]
    if boxes.numel() == 0:
        return boxes, scores

    keep = (scores >= conf_thr) & torch.tensor(
        [int(l.item()) in vehicle_class_ids for l in labels],
        dtype=torch.bool, device=scores.device,
    )
    boxes, scores = boxes[keep], scores[keep]

    if ignored_ltwh and boxes.numel() > 0:
        valid = []
        for b in boxes:
            bx1, by1, bx2, by2 = b.cpu().tolist()
            bltwh = (bx1, by1, bx2-bx1, by2-by1)
            max_iof = max(_iof_ltwh(bltwh, ig) for ig in ignored_ltwh) if ignored_ltwh else 0.
            valid.append(max_iof <= IOF_THR)
        mask = torch.tensor(valid, dtype=torch.bool, device=boxes.device)
        boxes, scores = boxes[mask], scores[mask]

    if len(boxes) > max_n:
        boxes, scores = boxes[:max_n], scores[:max_n]
    return boxes, scores


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiDatasetFrames(torch.utils.data.Dataset):
    """Load frames from any dataset using the DatasetAdapter interface."""

    def __init__(self, adapter, data_root: Path, video_list: List[str], transform):
        self.transform     = transform
        self.adapter       = adapter
        self.data_root     = data_root
        self.gt_cache: Dict[str, tuple] = {}
        self.samples: List[dict] = []

        frame_glob = adapter.get_frame_glob()
        for video in tqdm(video_list, desc="Loading GT"):
            try:
                gt_data, ignored = adapter.load_gt(video, data_root)
            except Exception as e:
                print(f"  [SKIP] {video}: GT error: {e}")
                continue
            # Cap ignored regions for performance (UAVDT has per-frame ignore boxes)
            if len(ignored) > 20:
                ignored = ignored[:20]
            self.gt_cache[video] = (gt_data, ignored)

            frame_dir = adapter.get_frame_dir(video, data_root)
            if not frame_dir.exists():
                print(f"  [SKIP] {video}: frame dir not found: {frame_dir}")
                continue
            for img_path in sorted(frame_dir.glob(frame_glob)):
                stem = img_path.stem.lstrip("img").lstrip("0") or "0"
                try:
                    fid = int(stem)
                except ValueError:
                    continue
                self.samples.append({
                    "img_path": img_path,
                    "video": video,
                    "frame_id": fid,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        ow, oh = img.size
        t = self.transform(img)
        nh, nw = t.shape[-2:]
        sx, sy = nw / float(ow), nh / float(oh)

        gt_data, ignored = self.gt_cache.get(s["video"], ({}, []))
        frame_objs = gt_data.get(s["frame_id"], [])

        gt_ltwh = [obj["bbox"] for obj in frame_objs]
        gt_xyxy = [(l*sx, t_*sy, (l+w)*sx, (t_+h)*sy) for (l, t_, w, h) in gt_ltwh]
        gt_t = torch.tensor(gt_xyxy, dtype=torch.float32)

        ign_ltwh = ignored
        ign_xyxy = [(l*sx, t_*sy, (l+w)*sx, (t_+h)*sy) for (l, t_, w, h) in ign_ltwh]
        ign_t = torch.tensor(ign_xyxy, dtype=torch.float32)

        return t, {"gt": gt_t, "ignored": ign_t, "ign_ltwh": ign_ltwh}, s


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


# ---------------------------------------------------------------------------
# Training (GFL)
# ---------------------------------------------------------------------------
def _make_gfl_label(box_xyxy, gt_xyxy_list, thr=MATCH_IOU_THR) -> float:
    if not gt_xyxy_list:
        return 0.
    bx1, by1, bx2, by2 = box_xyxy
    bltwh = (bx1, by1, bx2-bx1, by2-by1)
    iou = max(_iou_ltwh(bltwh, (gx1, gy1, gx2-gx1, gy2-gy1))
              for gx1, gy1, gx2, gy2 in gt_xyxy_list)
    return float(iou) if iou >= thr else 0.


def train_epoch_gfl(model, det_model, loader, optimizer, epoch, veh_ids):
    model.train()
    losses = []
    for imgs, targets, _ in tqdm(loader, desc=f"GFL epoch {epoch}"):
        imgs_t = torch.stack([i.to(DEVICE) for i in imgs])
        shapes = [tuple(i.shape[-2:]) for i in imgs]
        with torch.no_grad():
            outs = det_model(imgs_t)

        boxes_list, label_list = [], []
        for i, out in enumerate(outs):
            ign_ltwh = targets[i]["ign_ltwh"]
            b, _ = _filter_detections(out, ign_ltwh, CONF_THR, MAX_BOXES, veh_ids)
            boxes_list.append(b.detach())
            if len(b):
                gt_xyxy = targets[i]["gt"].tolist()
                lbls = torch.tensor(
                    [_make_gfl_label(b[k].cpu().tolist(), gt_xyxy) for k in range(len(b))],
                    dtype=torch.float32, device=DEVICE,
                )
                label_list.append(lbls)
            else:
                label_list.append(None)

        valid_labels = [l for l in label_list if l is not None]
        if not valid_labels:
            continue
        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        y = torch.cat(valid_labels, dim=0)
        if roi.shape[0] != y.shape[0]:
            continue
        loss = _quality_focal_loss(model(roi), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float('nan')


def train_epoch_iounet(model, det_model, loader, optimizer, epoch, veh_ids):
    model.train()
    losses = []
    for imgs, targets, _ in tqdm(loader, desc=f"IoUNet epoch {epoch}"):
        imgs_t = torch.stack([i.to(DEVICE) for i in imgs])
        shapes = [tuple(i.shape[-2:]) for i in imgs]
        with torch.no_grad():
            outs = det_model(imgs_t)

        boxes_list, label_list = [], []
        for i, out in enumerate(outs):
            ign_ltwh = targets[i]["ign_ltwh"]
            b, _ = _filter_detections(out, ign_ltwh, CONF_THR, MAX_BOXES, veh_ids)
            boxes_list.append(b.detach())
            if len(b):
                gt_xyxy = targets[i]["gt"].tolist()
                gt_ltwh = [(gx1, gy1, gx2-gx1, gy2-gy1) for gx1, gy1, gx2, gy2 in gt_xyxy]
                lbls = torch.tensor(
                    [_max_iou_to_gt(b[k].cpu().tolist(), gt_ltwh) for k in range(len(b))],
                    dtype=torch.float32, device=DEVICE,
                )
                label_list.append(lbls)
            else:
                label_list.append(None)

        valid_labels = [l for l in label_list if l is not None]
        if not valid_labels:
            continue
        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        y = torch.cat(valid_labels, dim=0)
        if roi.shape[0] != y.shape[0]:
            continue
        loss = F.smooth_l1_loss(model(roi), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float('nan')


# ---------------------------------------------------------------------------
# Inference → per-frame scores
# ---------------------------------------------------------------------------
@torch.no_grad()
def inference_scores(method, model, det_model, loader, veh_ids, out_dir: Path):
    model.eval()
    rows_by_video: Dict[str, List[dict]] = {}

    for imgs, targets, metas in tqdm(loader, desc="Inference"):
        imgs_t = torch.stack([i.to(DEVICE) for i in imgs])
        shapes = [tuple(i.shape[-2:]) for i in imgs]
        outs   = det_model(imgs_t)

        boxes_list, meta_boxes = [], []
        for i, out in enumerate(outs):
            ign_ltwh = targets[i]["ign_ltwh"]
            b, _ = _filter_detections(out, ign_ltwh, CONF_THR, MAX_BOXES, veh_ids)
            boxes_list.append(b.detach())
            if len(b):
                for _ in range(len(b)):
                    meta_boxes.append({"video": metas[i]["video"],
                                       "frame_id": metas[i]["frame_id"]})

        if not meta_boxes:
            for m in metas:
                rows_by_video.setdefault(m["video"], []).append(
                    {"frame_id": m["frame_id"], "score_mean": 0., "score_max": 0., "score_top3": 0.})
            continue

        roi = _roi_features(det_model, imgs_t, boxes_list, shapes)
        if method == "gfl":
            pred = torch.sigmoid(model(roi)).cpu().numpy()
        else:
            pred = model(roi).clamp(0., 1.).cpu().numpy()

        assert len(pred) == len(meta_boxes)
        box_df = pd.DataFrame(meta_boxes)
        box_df["score"] = pred

        detected_fids = set()
        for (video, fid), g in box_df.groupby(["video", "frame_id"]):
            detected_fids.add(fid)
            q = g["score"].values
            top3 = float(np.mean(np.sort(q)[::-1][:min(3, len(q))]))
            rows_by_video.setdefault(video, []).append({
                "frame_id": int(fid),
                "score_mean": float(np.mean(q)),
                "score_max":  float(np.max(q)),
                "score_top3": top3,
            })

        for m in metas:
            if m["frame_id"] not in detected_fids:
                rows_by_video.setdefault(m["video"], []).append(
                    {"frame_id": m["frame_id"], "score_mean": 0., "score_max": 0., "score_top3": 0.})

    # Save per-video CSVs
    if method == "gfl":
        col_map = {"score_mean": "gfl_mean", "score_max": "gfl_max", "score_top3": "gfl_top3"}
    else:
        col_map = {"score_mean": "iou_net_mean", "score_max": "iou_net_max", "score_top3": "iou_net_top3"}

    out_dir.mkdir(parents=True, exist_ok=True)
    for video, rows in rows_by_video.items():
        df = pd.DataFrame(rows).rename(columns=col_map)
        df = df.sort_values("frame_id").reset_index(drop=True)
        out_path = out_dir / f"{video}.csv"
        df[["frame_id"] + list(col_map.values())].to_csv(
            out_path, index=False, float_format="%.6f")
        print(f"  {video}: {len(df)} frames → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-dataset GFL/IoUNet training")
    parser.add_argument("--dataset", required=True,
                        choices=["ua-detrac", "uavdt", "mot17", "imagenet-vid"])
    parser.add_argument("--method",  required=True, choices=["gfl", "iounet"])
    parser.add_argument("--splits-file", type=str,
                        default=str(SCRIPT_DIR / "baseline_eval" / "video_splits.json"))
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weights",    type=str,   default=None,
                        help="Pre-trained .pth weights (skips training)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training, only run inference (requires --weights)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Dataset-specific vehicle class IDs
    if args.dataset == "mot17":
        veh_ids = PERSON_COCO
    else:
        veh_ids = VEHICLE_COCO

    with open(args.splits_file) as f:
        splits = json.load(f)

    ds_splits = splits[args.dataset]
    train_vids = ds_splits["train"]
    test_vids  = ds_splits["test"]

    adapter   = get_adapter(args.dataset)
    data_root = DATA_ROOTS[args.dataset]
    score_dir_name = "gfl_scores" if args.method == "gfl" else "iounet_scores"

    out_dir = Path(args.output_dir) if args.output_dir else \
              SCRIPT_DIR / score_dir_name / MODEL_WEAK / CONF_LABEL / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # MOT17 FRCNN → DPM mapping for image/GT loading
    def _resolve_video(v):
        if "FRCNN" in v:
            return v.replace("FRCNN", "DPM")
        return v

    if args.dataset == "mot17":
        train_vids_resolved = [_resolve_video(v) for v in train_vids]
        test_vids_resolved  = [_resolve_video(v) for v in test_vids]
    else:
        train_vids_resolved = train_vids
        test_vids_resolved  = test_vids

    print(f"Device:      {DEVICE}")
    print(f"Dataset:     {args.dataset}")
    print(f"Method:      {args.method}")
    print(f"Train:       {len(train_vids_resolved)} videos")
    print(f"Test:        {len(test_vids_resolved)} videos")
    print(f"Output dir:  {out_dir}")

    det_model, transform = _load_detector()
    roi_dim = _infer_roi_dim(det_model)
    print(f"RoI dim:     {roi_dim}")

    if args.method == "gfl":
        predictor = GFLPredictor(roi_dim).to(DEVICE)
    else:
        predictor = IoUPredictor(roi_dim).to(DEVICE)

    # Load weights if provided (with key remapping from fc1/fc2/fc3 → fc1/fc2/fc3 already)
    if args.weights:
        ckpt = torch.load(args.weights, map_location=DEVICE)
        predictor.load_state_dict(ckpt, strict=False)
        print(f"Loaded weights: {args.weights}")

    if not args.inference_only and not args.weights:
        # Train
        train_ds = MultiDatasetFrames(adapter, data_root, train_vids_resolved, transform)
        print(f"\nTrain samples: {len(train_ds)}")
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=_collate,
        )
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
        print(f"\n=== Training {args.method.upper()} on {args.dataset} ===")
        for ep in range(1, args.epochs + 1):
            if args.method == "gfl":
                loss = train_epoch_gfl(predictor, det_model, train_loader, optimizer, ep, veh_ids)
            else:
                loss = train_epoch_iounet(predictor, det_model, train_loader, optimizer, ep, veh_ids)
            print(f"  Epoch {ep}/{args.epochs}  loss={loss:.5f}")

        ckpt_path = out_dir / f"{args.method}.pth"
        torch.save(predictor.state_dict(), ckpt_path)
        print(f"Saved model: {ckpt_path}")

    # Inference on test videos
    print(f"\n=== Inference on {args.dataset} test split ===")
    test_ds = MultiDatasetFrames(adapter, data_root, test_vids_resolved, transform)
    print(f"Test samples: {len(test_ds)}")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate,
    )

    # Override video names in inference to use original (FRCNN) names
    if args.dataset == "mot17":
        name_map = {_resolve_video(v): v for v in test_vids}
        for s in test_ds.samples:
            if s["video"] in name_map:
                s["video"] = name_map[s["video"]]

    inference_scores(args.method, predictor, det_model, test_loader, veh_ids, out_dir)
    print(f"\nDone. Scores saved to: {out_dir}")


if __name__ == "__main__":
    main()
