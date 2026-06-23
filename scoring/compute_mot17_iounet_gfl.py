#!/usr/bin/env python3
"""
Train IoU-Net and GFL quality estimators on MOT17 and produce per-frame score CSVs.

Both models use RoI features from FasterRCNN-MobileNetV3-Large-320-FPN (COCO pretrained,
auto-downloaded). Trained on 60% of MOT17 train sequences (grouped by base video, seed=42),
evaluated on remaining 40% (val split = MOT17-11-FRCNN and MOT17-13-FRCNN).

No pre-trained checkpoints needed beyond TorchVision COCO weights.

Dataset layout (set --mot17-root to override):
    data/mot17/train/
        MOT17-02-FRCNN/
            img1/000001.jpg ...
            gt/gt.txt
        MOT17-04-FRCNN/ ...
        MOT17-05-FRCNN/ ...
        MOT17-09-FRCNN/ ...
        MOT17-10-FRCNN/ ...
        MOT17-11-FRCNN/ ...
        MOT17-13-FRCNN/ ...

Output (per-seq CSVs consumed by compute_mot17_routing.py):
    scores/iounet/mot17/{seq}.csv  — columns: frame_id, iou_net_mean, iou_net_max, iou_net_top3
    scores/gfl/mot17/{seq}.csv     — columns: frame_id, gfl_mean, gfl_max, gfl_top3

Weights saved for reuse:
    weights/neural/iou_net_mot17.pth
    weights/neural/gfl_mot17.pth

Usage:
    python scoring/compute_mot17_iounet_gfl.py
    python scoring/compute_mot17_iounet_gfl.py --method iounet
    python scoring/compute_mot17_iounet_gfl.py --method gfl
    python scoring/compute_mot17_iounet_gfl.py --mot17-root /data/MOT17/train
    python scoring/compute_mot17_iounet_gfl.py --skip-existing   # skip already-scored seqs
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.ops import box_iou
from tqdm import tqdm

ROOT = Path(__file__).parent.parent

DEFAULT_MOT17_ROOT = ROOT / "data" / "mot17" / "train"
SCORES_BASE        = ROOT / "scores"
WEIGHTS_DIR        = ROOT / "weights" / "neural"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_RATIO    = 0.6
RANDOM_SEED    = 42
EPOCHS         = 5
BATCH_SIZE     = 8
LR             = 1e-4
CONF_THR       = 0.1          # detection confidence threshold
MAX_BOXES      = 100          # max predicted boxes per frame during inference
PERSON_CLASS   = 1            # COCO label id for person

# Fixed input resize for MOT17. MOT17 has mixed native resolutions, so all
# frames are resized to this for batched inference.
MOT17_RESIZE_W = 640
MOT17_RESIZE_H = 384

# IoU-Net GT jitter config
JITTERS_PER_GT      = 16
MAX_JITTER_PER_IMG  = 256
JITTER_CENTER_STD   = 0.20
JITTER_SCALE_STD    = 0.25
JITTER_MIN_IOU      = 0.05

# GFL config
MATCH_IOU_THR = 0.5
QFL_BETA      = 2.0

# Train seqs (base videos) — TRAIN_RATIO=0.6 on 7 base videos
# 7 * 0.6 = 4.2 → 4 train, 3 val  (sorted, seed=42)
# train=02,04,05,09  val=10,11,13


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _seed(s: int = RANDOM_SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# MOT17 GT parsing
# ---------------------------------------------------------------------------
def _load_gt(seq_dir: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    gt_file = seq_dir / "gt" / "gt.txt"
    frame_gt: Dict[int, List] = {}
    with open(gt_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            fid    = int(float(parts[0]))
            x, y   = float(parts[2]), float(parts[3])
            w, h   = float(parts[4]), float(parts[5])
            mark   = float(parts[6])
            cls_id = int(float(parts[7]))
            if mark <= 0 or cls_id != 1 or w <= 0 or h <= 0:
                continue
            frame_gt.setdefault(fid, []).append((x, y, x + w, y + h))
    return frame_gt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MOT17Dataset(Dataset):
    def __init__(self, seq_dirs: List[Path], transform):
        self.transform = transform
        self.samples: List[Dict] = []
        for sd in seq_dirs:
            gt = _load_gt(sd)
            imgs = sorted((sd / "img1").glob("*.jpg"))
            for img_p in imgs:
                fid = int(re.sub(r"\D", "", img_p.stem))
                self.samples.append({
                    "img_path": img_p,
                    "frame_idx": fid,
                    "video": sd.name.lower(),
                    "gt_boxes": gt.get(fid, []),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        # Fixed resize so batch_size > 1 works on MOT17, whose sequences have
        # mixed native resolutions (1920x1080 and 640x480). GT boxes are scaled
        # to the resized frame.
        img = img.resize((MOT17_RESIZE_W, MOT17_RESIZE_H), Image.BILINEAR)
        img = self.transform(img)
        sx = MOT17_RESIZE_W / orig_w
        sy = MOT17_RESIZE_H / orig_h
        if s["gt_boxes"]:
            gt = torch.tensor(s["gt_boxes"], dtype=torch.float32)
            gt = gt * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        else:
            gt = torch.zeros((0, 4), dtype=torch.float32)
        return img, gt, s


def _collate(batch):
    imgs, gts, metas = zip(*batch)
    return list(imgs), list(gts), list(metas)


# ---------------------------------------------------------------------------
# ROI feature extraction
# ---------------------------------------------------------------------------
def _get_roi_features(detector, imgs_t: torch.Tensor,
                      boxes_list: List[torch.Tensor],
                      img_shapes: List[Tuple[int, int]]) -> torch.Tensor:
    feats = detector.backbone(imgs_t)
    return detector.roi_heads.box_roi_pool(feats, boxes_list, img_shapes)


def _infer_roi_dim(detector, device: str, transform) -> int:
    dummy = torch.zeros(1, 3, 320, 320).to(device)
    dummy_box = [torch.tensor([[0., 0., 100., 100.]], device=device)]
    feats = detector.backbone(dummy)
    roi = detector.roi_heads.box_roi_pool(feats, dummy_box, [(320, 320)])
    return int(roi.flatten(1).shape[1])


# ---------------------------------------------------------------------------
# IoU-Net model
# ---------------------------------------------------------------------------
class IoUPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)


# ---------------------------------------------------------------------------
# GFL model
# ---------------------------------------------------------------------------
class GFLPredictor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze(1)


# ---------------------------------------------------------------------------
# Jitter box generation for IoU-Net training
# ---------------------------------------------------------------------------
def _jitter_boxes(gt_boxes: torch.Tensor, img_shape: Tuple[int, int],
                  device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if gt_boxes.numel() == 0:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,),   device=device))
    H, W = img_shape
    gt = gt_boxes.to(device).float()
    parts = []
    for b in gt:
        x1, y1, x2, y2 = b
        bw = (x2 - x1).clamp(min=2.)
        bh = (y2 - y1).clamp(min=2.)
        cx = x1 + .5 * bw;  cy = y1 + .5 * bh
        dx = torch.randn(JITTERS_PER_GT, device=device) * JITTER_CENTER_STD * bw
        dy = torch.randn(JITTERS_PER_GT, device=device) * JITTER_CENTER_STD * bh
        sw = torch.exp(torch.randn(JITTERS_PER_GT, device=device) * JITTER_SCALE_STD)
        sh = torch.exp(torch.randn(JITTERS_PER_GT, device=device) * JITTER_SCALE_STD)
        nw = (bw * sw).clamp(2., float(W));  nh = (bh * sh).clamp(2., float(H))
        jx1 = (cx + dx - .5 * nw).clamp(0, W - 1)
        jy1 = (cy + dy - .5 * nh).clamp(0, H - 1)
        jx2 = (cx + dx + .5 * nw).clamp(0, W - 1)
        jy2 = (cy + dy + .5 * nh).clamp(0, H - 1)
        jitter = torch.stack([jx1, jy1, jx2, jy2], 1)
        # include the GT box itself as IoU=1 sample
        parts.append(torch.cat([jitter, b.unsqueeze(0)], 0))
    boxes = torch.cat(parts, 0)
    valid = (boxes[:, 2] > boxes[:, 0] + 1.) & (boxes[:, 3] > boxes[:, 1] + 1.)
    boxes = boxes[valid]
    if boxes.numel() == 0:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,),   device=device))
    ious = box_iou(boxes, gt).max(1).values
    keep = ious >= JITTER_MIN_IOU
    boxes, ious = boxes[keep], ious[keep]
    if boxes.shape[0] > MAX_JITTER_PER_IMG:
        idx = torch.randperm(boxes.shape[0], device=device)[:MAX_JITTER_PER_IMG]
        boxes, ious = boxes[idx], ious[idx]
    return boxes.detach(), ious.detach()


# ---------------------------------------------------------------------------
# GFL quality focal loss
# ---------------------------------------------------------------------------
def _qfl(pred: torch.Tensor, target: torch.Tensor, beta: float = QFL_BETA) -> torch.Tensor:
    pt = torch.abs(target - pred)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    return (pt ** beta * bce).mean()


# ---------------------------------------------------------------------------
# Sequence split: 60/40, grouped by base video, seed=42
# ---------------------------------------------------------------------------
def _split_seqs(seq_dirs: List[Path]) -> Tuple[List[Path], List[Path]]:
    base_to_seqs: Dict[str, List[Path]] = {}
    for sd in seq_dirs:
        base = "-".join(sd.name.split("-")[:2])
        base_to_seqs.setdefault(base, []).append(sd)
    bases = sorted(base_to_seqs.keys())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(bases)
    n_train = max(1, int(round(len(bases) * TRAIN_RATIO)))
    train_bases = set(bases[:n_train])
    train_seqs = [sd for b in train_bases for sd in base_to_seqs[b]]
    val_seqs   = [sd for b in bases[n_train:] for sd in base_to_seqs.get(b, [])]
    return sorted(train_seqs), sorted(val_seqs)


# ---------------------------------------------------------------------------
# Detector predicted boxes (for inference)
# ---------------------------------------------------------------------------
def _detect_boxes(outputs, device: str) -> List[torch.Tensor]:
    boxes_list = []
    for out in outputs:
        mask = (out["labels"] == PERSON_CLASS) & (out["scores"] >= CONF_THR)
        b = out["boxes"][mask][:MAX_BOXES].to(device)
        boxes_list.append(b)
    return boxes_list


# ---------------------------------------------------------------------------
# Training: IoU-Net
# ---------------------------------------------------------------------------
def _train_iounet(detector, model: IoUPredictor, loader: DataLoader,
                  device: str, epochs: int):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train(); detector.eval()
    for ep in range(1, epochs + 1):
        losses = []
        for imgs, gts, _ in tqdm(loader, desc=f"IoU-Net epoch {ep}/{epochs}"):
            imgs_t = torch.stack([i.to(device) for i in imgs])
            shapes = [tuple(i.shape[-2:]) for i in imgs]
            boxes_list, labels_list = [], []
            for i, gt in enumerate(gts):
                jb, jl = _jitter_boxes(gt, shapes[i], device)
                boxes_list.append(jb)
                if jl.numel() > 0:
                    labels_list.append(jl)
            if not labels_list:
                continue
            y = torch.cat(labels_list)
            roi = _get_roi_features(detector, imgs_t, boxes_list, shapes)
            if roi.shape[0] != y.shape[0]:
                continue
            loss = F.smooth_l1_loss(model(roi), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        print(f"  epoch {ep} avg loss: {np.mean(losses):.5f}")


# ---------------------------------------------------------------------------
# Training: GFL
# ---------------------------------------------------------------------------
def _train_gfl(detector, model: GFLPredictor, loader: DataLoader,
               device: str, epochs: int):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train(); detector.eval()
    for ep in range(1, epochs + 1):
        losses = []
        for imgs, gts, _ in tqdm(loader, desc=f"GFL epoch {ep}/{epochs}"):
            imgs_t = torch.stack([i.to(device) for i in imgs])
            shapes = [tuple(i.shape[-2:]) for i in imgs]
            with torch.no_grad():
                outputs = detector(imgs_t)
            boxes_list = _detect_boxes(outputs, device)
            all_boxes, all_labels = [], []
            for i, (bx, gt) in enumerate(zip(boxes_list, gts)):
                if bx.numel() == 0:
                    continue
                gt_t = gt.to(device).float()
                if gt_t.numel() == 0:
                    labels = torch.zeros(len(bx), device=device)
                else:
                    ious = box_iou(bx, gt_t).max(1).values
                    labels = torch.where(ious >= MATCH_IOU_THR, ious,
                                         torch.zeros_like(ious))
                all_boxes.append(bx); all_labels.append(labels)
            if not all_boxes:
                continue
            all_bx = torch.cat(all_boxes)
            y = torch.cat(all_labels)
            # rebuild boxes_list in batch order for ROI pool
            roi = _get_roi_features(detector, imgs_t, boxes_list, shapes)
            # roi rows correspond to all non-empty boxes in batch order
            nonempty = [bx for bx in boxes_list if bx.numel() > 0]
            expected = sum(len(b) for b in nonempty)
            if roi.shape[0] != expected or roi.shape[0] != y.shape[0]:
                continue
            loss = _qfl(model(roi), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        print(f"  epoch {ep} avg loss: {np.mean(losses):.5f}")


# ---------------------------------------------------------------------------
# Inference → per-frame aggregation → save CSVs
# ---------------------------------------------------------------------------
def _infer_iounet(detector, model: IoUPredictor,
                  seq_dirs: List[Path], transform, device: str,
                  out_dir: Path, skip_existing: bool):
    model.eval(); detector.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sd in seq_dirs:
            out_path = out_dir / f"{sd.name}.csv"
            if skip_existing and out_path.exists():
                print(f"  [SKIP] {sd.name}")
                continue
            imgs = sorted((sd / "img1").glob("*.jpg"))
            rows = []
            for i in range(0, len(imgs), BATCH_SIZE):
                batch = imgs[i:i + BATCH_SIZE]
                imgs_t = torch.stack([
                    transform(Image.open(p).convert("RGB").resize(
                        (MOT17_RESIZE_W, MOT17_RESIZE_H), Image.BILINEAR))
                    for p in batch]).to(device)
                shapes = [tuple(t.shape[-2:]) for t in imgs_t]
                outputs = detector(imgs_t)
                boxes_list = _detect_boxes(outputs, device)
                roi = _get_roi_features(detector, imgs_t, boxes_list, shapes)
                # split roi back per image
                sizes = [len(b) for b in boxes_list]
                split_roi = torch.split(roi, sizes, dim=0) if sum(sizes) > 0 else \
                            [torch.empty((0, roi.shape[1]), device=device)] * len(batch)
                for img_p, bx, scores_out, r in zip(batch, boxes_list,
                                                     [o["scores"] for o in outputs],
                                                     split_roi):
                    fid = int(re.sub(r"\D", "", img_p.stem))
                    if bx.numel() == 0:
                        rows.append({"frame_id": fid, "iou_net_mean": 0.,
                                     "iou_net_max": 0., "iou_net_top3": 0.,
                                     "iou_net_conf_weighted_mean": 0.})
                        continue
                    # boxes_list (bx) and scores_out are already filtered to
                    # person + conf>=CONF_THR in _detect_boxes; take the first
                    # len(bx) scores which correspond row-for-row to bx.
                    conf = scores_out[:len(bx)].cpu().numpy()
                    pred = torch.clamp(model(r), 0., 1.).cpu().numpy()
                    conf_sum = float(conf.sum())
                    cw = float((pred * conf).sum() / conf_sum) if conf_sum > 1e-12 \
                         else float(pred.mean())
                    top3 = float(np.sort(pred)[::-1][:3].mean())
                    rows.append({
                        "frame_id":       fid,
                        "iou_net_mean":   float(pred.mean()),
                        "iou_net_max":    float(pred.max()),
                        "iou_net_top3":   top3,
                        # conf-weighted mean: the routing column used by the
                        # reference MOT17 pipeline (matches paper Tab 2 pairwise).
                        "iou_net_conf_weighted_mean": cw,
                    })
            pd.DataFrame(rows).sort_values("frame_id").to_csv(
                out_path, index=False, float_format="%.6f")
            print(f"  {sd.name}: {len(rows)} frames → {out_path.name}")


def _infer_gfl(detector, model: GFLPredictor,
               seq_dirs: List[Path], transform, device: str,
               out_dir: Path, skip_existing: bool):
    model.eval(); detector.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sd in seq_dirs:
            out_path = out_dir / f"{sd.name}.csv"
            if skip_existing and out_path.exists():
                print(f"  [SKIP] {sd.name}")
                continue
            imgs = sorted((sd / "img1").glob("*.jpg"))
            rows = []
            for i in range(0, len(imgs), BATCH_SIZE):
                batch = imgs[i:i + BATCH_SIZE]
                imgs_t = torch.stack([
                    transform(Image.open(p).convert("RGB").resize(
                        (MOT17_RESIZE_W, MOT17_RESIZE_H), Image.BILINEAR))
                    for p in batch]).to(device)
                shapes = [tuple(t.shape[-2:]) for t in imgs_t]
                outputs = detector(imgs_t)
                boxes_list = _detect_boxes(outputs, device)
                roi = _get_roi_features(detector, imgs_t, boxes_list, shapes)
                sizes = [len(b) for b in boxes_list]
                split_roi = torch.split(roi, sizes, dim=0) if sum(sizes) > 0 else \
                            [torch.empty((0, roi.shape[1] if roi.numel() > 0 else 1),
                                         device=device)] * len(batch)
                for img_p, bx, scores_out, r in zip(batch, boxes_list,
                                                     [o["scores"] for o in outputs],
                                                     split_roi):
                    fid = int(re.sub(r"\D", "", img_p.stem))
                    if r.numel() == 0:
                        rows.append({"frame_id": fid, "gfl_mean": 0.,
                                     "gfl_max": 0., "gfl_top3": 0.,
                                     "gfl_final_score_mean": 0.})
                        continue
                    pred = model(r).cpu().numpy()           # GFL quality in [0,1]
                    conf = scores_out[:len(bx)].cpu().numpy()
                    top3 = float(np.sort(pred)[::-1][:3].mean())
                    # gfl_final_score = conf * quality (per box); routing uses its
                    # per-frame mean (matches reference MOT17 pipeline + paper Tab 2).
                    final = conf * pred
                    rows.append({
                        "frame_id": fid,
                        "gfl_mean": float(pred.mean()),
                        "gfl_max":  float(pred.max()),
                        "gfl_top3": top3,
                        "gfl_final_score_mean": float(final.mean()),
                    })
            pd.DataFrame(rows).sort_values("frame_id").to_csv(
                out_path, index=False, float_format="%.6f")
            print(f"  {sd.name}: {len(rows)} frames → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train IoU-Net/GFL on MOT17 and score all sequences")
    parser.add_argument("--method", choices=["iounet", "gfl", "both"], default="both")
    parser.add_argument("--mot17-root", type=str, default=str(DEFAULT_MOT17_ROOT))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip sequences whose score CSV already exists")
    parser.add_argument("--retrain", action="store_true",
                        help="Re-train even if weights already exist")
    args = parser.parse_args()

    _seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mot17_root = Path(args.mot17_root)
    if not mot17_root.is_dir():
        raise SystemExit(
            f"MOT17 root not found: {mot17_root}\n"
            "Set --mot17-root to the directory containing MOT17-*-FRCNN/ folders.\n"
            "Run compute_mot17_detections.py first if you also need XML detection files."
        )

    # Discover FRCNN sequences
    all_seqs = sorted(
        d for d in mot17_root.iterdir()
        if d.is_dir() and d.name.endswith("-FRCNN") and (d / "img1").is_dir()
    )
    if not all_seqs:
        raise SystemExit(f"No MOT17-*-FRCNN sequences found under {mot17_root}")
    print(f"Sequences: {[d.name for d in all_seqs]}")

    train_seqs, val_seqs = _split_seqs(all_seqs)
    print(f"Train: {[d.name for d in train_seqs]}")
    print(f"Val:   {[d.name for d in val_seqs]}")

    # Load TorchVision detector (COCO pretrained, auto-download).
    # NOTE: despite the "320" label used elsewhere, the FULL-resolution mobilenet
    # (min 800/max 1333) is used for RoI features, not the literal _320_fpn model.
    weights   = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    transform = weights.transforms()
    detector  = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(device)
    detector.eval()

    roi_dim = _infer_roi_dim(detector, device, transform)
    print(f"ROI feature dim: {roi_dim}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["iounet", "gfl"] if args.method == "both" else [args.method]

    for method in methods:
        print(f"\n{'='*60}\nMethod: {method.upper()}\n{'='*60}")
        ckpt = WEIGHTS_DIR / (f"iou_net_mot17.pth" if method == "iounet"
                              else "gfl_mot17.pth")
        out_dir = SCORES_BASE / method / "mot17"

        if method == "iounet":
            model = IoUPredictor(roi_dim).to(device)
        else:
            model = GFLPredictor(roi_dim).to(device)

        if ckpt.exists() and not args.retrain:
            print(f"Loading existing weights: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print("Training...")
            train_ds = MOT17Dataset(train_seqs, transform)
            loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, collate_fn=_collate)
            if method == "iounet":
                _train_iounet(detector, model, loader, device, args.epochs)
            else:
                _train_gfl(detector, model, loader, device, args.epochs)
            torch.save(model.state_dict(), ckpt)
            print(f"Saved weights: {ckpt}")

        print(f"\nScoring all {len(all_seqs)} sequences...")
        if method == "iounet":
            _infer_iounet(detector, model, all_seqs, transform, device,
                          out_dir, args.skip_existing)
        else:
            _infer_gfl(detector, model, all_seqs, transform, device,
                       out_dir, args.skip_existing)

    print("\nDone.")


if __name__ == "__main__":
    main()
