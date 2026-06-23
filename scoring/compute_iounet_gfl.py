#!/usr/bin/env python3
"""
Run IoU-Net and GFL inference across datasets to produce per-frame quality scores.

Both methods use RoI-pooled features from the FasterRCNN-MobileNetV3 backbone.
Trained on UA-DETRAC train split; evaluated zero-shot on UAVDT and MOT17.

Prerequisites:
  - Trained weights in weights/neural/iou_net.pth and weights/neural/gfl.pth
    (produced by training/train_iounet.py and training/train_gfl.py)
  - Detection .txt files in data/{DATASET}/detections/FASTERRCNN-MOBILENET-V3-LARGE-FPN/
  - Raw video frames (JPEG) in data/{DATASET}/frames/ for backbone feature extraction

Output layout:
    scores/iounet/{dataset}/{video}.csv  — columns: frame_id, iou_net_mean, iou_net_max, iou_net_top3
    scores/gfl/{dataset}/{video}.csv     — columns: frame_id, gfl_mean, gfl_max, gfl_top3

Usage:
    python scoring/compute_iounet_gfl.py --method iounet
    python scoring/compute_iounet_gfl.py --method gfl
    python scoring/compute_iounet_gfl.py --method both --datasets ua-detrac uavdt mot17
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "estimator"))
sys.path.insert(0, str(ROOT / "dataset"))

from iounet_inference import IoUPredictor, IoUNetEstimator, ROI_DIM, CONF_THR, CONF_THR_MOT17, PERSON_IDS, VEHICLE_IDS
from gfl_inference import GFLEstimator
from adapters import get_adapter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IOUNET_CKPT       = ROOT / "weights" / "neural" / "iou_net.pth"
GFL_CKPT          = ROOT / "weights" / "neural" / "gfl.pth"
IOUNET_CKPT_MOT17 = ROOT / "weights" / "neural" / "iou_net_mot17.pth"
GFL_CKPT_MOT17    = ROOT / "weights" / "neural" / "gfl_mot17.pth"
SPLITS_FILE       = ROOT / "dataset" / "video_splits.json"
SCORES_BASE       = ROOT / "scores"

# Per-dataset configuration:
#   UA-DETRAC / UAVDT: non-320 FPN, vehicle classes, CONF_THR=0.5
#   MOT17:             320-FPN, person class, CONF_THR=0.1, separate weights
MODEL_WEAK       = "FASTERRCNN-MOBILENET-V3-LARGE-FPN"
MODEL_WEAK_MOT17 = "FASTERRCNN-MOBILENET-V3-LARGE-320-FPN"
IOF_THR          = 0.5

# Dataset roots (download datasets into <repo>/data/; see detection/README.md).
DATA_ROOTS = {
    "ua-detrac": ROOT / "data" / "UA-DETRAC",
    "uavdt":     ROOT / "data" / "UAVDT",
    "mot17":     ROOT / "data" / "MOT17",
}

VEHICLE_COCO = {3, 4, 6, 8}   # car, motorcycle, bus, truck
PERSON_COCO  = {1}             # person (MOT17)


def _valid_ids(dataset: str):
    return PERSON_COCO if dataset == "mot17" else VEHICLE_COCO


def _load_ignored_xyxy(adapter, video: str, data_root: Path):
    """Load the video's ignored regions as an (N,4) xyxy tensor.

    Detections whose IoF with any ignored region exceeds IOF_THR are filtered
    out before scoring. UA-DETRAC
    in particular has many ignored regions; omitting this would retain
    extra boxes (≈+2/frame), shifting per-frame mean scores and ranking.
    Adapter ignored regions are xywh; score_frame expects xyxy.
    """
    try:
        _, ignored_xywh = adapter.load_gt(video, data_root)
    except Exception:
        ignored_xywh = []
    if not ignored_xywh:
        return None
    xyxy = [[l, t, l + w, t + h] for (l, t, w, h) in ignored_xywh]
    return torch.tensor(xyxy, dtype=torch.float32)


def score_video_iounet(est: IoUNetEstimator, adapter, video: str,
                        data_root: Path, out_path: Path) -> bool:
    frame_dir = adapter.get_frame_dir(video, data_root)
    frames = sorted(frame_dir.glob(adapter.get_frame_glob()))
    if not frames:
        print(f"  [SKIP] {video}: no frames at {frame_dir}")
        return False

    ignored_xyxy = _load_ignored_xyxy(adapter, video, data_root)
    rows = []
    for img_path in tqdm(frames, desc=video, leave=False):
        fid = int(''.join(c for c in img_path.stem if c.isdigit()))
        img_t = est.transform(Image.open(img_path).convert("RGB"))
        scores = est.score_frame(img_t, ignored_xyxy=ignored_xyxy)
        rows.append({"frame_id": fid, **scores})

    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    return True


def score_video_gfl(est: GFLEstimator, adapter, video: str,
                     data_root: Path, out_path: Path) -> bool:
    frame_dir = adapter.get_frame_dir(video, data_root)
    frames = sorted(frame_dir.glob(adapter.get_frame_glob()))
    if not frames:
        print(f"  [SKIP] {video}: no frames at {frame_dir}")
        return False

    ignored_xyxy = _load_ignored_xyxy(adapter, video, data_root)
    rows = []
    for img_path in tqdm(frames, desc=video, leave=False):
        fid = int(''.join(c for c in img_path.stem if c.isdigit()))
        img_t = est.transform(Image.open(img_path).convert("RGB"))
        scores = est.score_frame(img_t, ignored_xyxy=ignored_xyxy)
        rows.append({"frame_id": fid, **scores})

    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.6f")
    return True


# ---------------------------------------------------------------------------
# MOT17: delegate to compute_mot17_iounet_gfl.py (trains from scratch)
# ---------------------------------------------------------------------------
MOT17_SCRIPT = Path(__file__).parent / "compute_mot17_iounet_gfl.py"


def run_mot17_iounet_gfl(method: str, mot17_root: Optional[str] = None):
    """Delegate MOT17 IoU-Net/GFL scoring to compute_mot17_iounet_gfl.py."""
    import subprocess
    cmd = [sys.executable, str(MOT17_SCRIPT), "--method", method]
    if mot17_root:
        cmd += ["--mot17-root", mot17_root]
    print(f"\n{method.upper()} — mot17 (train from scratch via compute_mot17_iounet_gfl.py)")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[WARN] compute_mot17_iounet_gfl.py exited with code {ret.returncode}")


def run_method(method: str, datasets: List[str], split: str, splits: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_base = SCORES_BASE / method

    for ds in datasets:
        # MOT17: train from scratch via dedicated script
        if ds == "mot17":
            run_mot17_iounet_gfl(method)
            continue

        # Select dataset-specific weights, conf_thr, class IDs, and model
        if ds == "mot17":
            ckpt      = IOUNET_CKPT_MOT17 if method == "iounet" else GFL_CKPT_MOT17
            conf_thr  = CONF_THR_MOT17
            class_ids = PERSON_IDS
            use_320   = True
        else:
            ckpt      = IOUNET_CKPT if method == "iounet" else GFL_CKPT
            conf_thr  = CONF_THR
            class_ids = VEHICLE_IDS
            use_320   = False

        if not ckpt.exists():
            print(f"[SKIP] {ds}: weights not found at {ckpt}")
            continue

        if method == "iounet":
            est = IoUNetEstimator(str(ckpt), device=device, conf_thr=conf_thr,
                                  class_ids=class_ids, use_320=use_320)
            score_fn = score_video_iounet
        else:
            est = GFLEstimator(str(ckpt), device=device, conf_thr=conf_thr,
                               class_ids=class_ids, use_320=use_320)
            score_fn = score_video_gfl

        adapter   = get_adapter(ds)
        data_root = DATA_ROOTS[ds]
        out_dir   = out_base / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_splits = splits.get(ds, {})
        videos = []
        if split in ("train", "both"):
            videos += ds_splits.get("train", [])
        if split in ("test", "both"):
            videos += ds_splits.get("test", [])

        print(f"\n{method.upper()} — {ds} (conf_thr={conf_thr}, use_320={use_320}, "
              f"{len(videos)} videos)")
        ok = 0
        for v in videos:
            out_path = out_dir / f"{v}.csv"
            if out_path.exists():
                print(f"  [SKIP] {v}: already exists")
                ok += 1
                continue
            ok += score_fn(est, adapter, v, data_root, out_path)
        print(f"Done: {ok}/{len(videos)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["iounet", "gfl", "both"], default="both")
    parser.add_argument("--datasets", nargs="+",
                        default=["ua-detrac", "uavdt", "mot17"],
                        choices=["ua-detrac", "uavdt", "mot17"])
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    methods = ["iounet", "gfl"] if args.method == "both" else [args.method]
    for m in methods:
        run_method(m, args.datasets, args.split, splits)


if __name__ == "__main__":
    main()
