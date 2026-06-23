#!/usr/bin/env python3
"""
Train the XGBoost difficulty predictor on UA-DETRAC (train split).

Derives a 30-feature sliding-window representation from the weak model's
detection output stream and regresses onto per-frame ∆mAP
(strong_mAP − weak_mAP).  An RF gatekeeper filters unreliable windows
before passing to XGBoost.

Training data: UA-DETRAC official train split
Zero-shot eval: UAVDT, MOT17 (run scoring/compute_xgboost.py)

Output:
    weights/xgboost/predictor.pkl   — trained DifficultyPredictor

Usage:
    python training/train_xgboost.py
    python training/train_xgboost.py --window 7 --horizon 1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "estimator"))
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT / "scoring"))

from predictor import DifficultyPredictor
from gt_parser import get_video_split

# UA-DETRAC dataset (download into <repo>/data/; see detection/README.md).
DATA_ROOT   = ROOT / "data" / "UA-DETRAC"
SPLITS_FILE = ROOT / "dataset" / "video_splits.json"
DERIVED_DIR = ROOT / "scores" / "derived"   # per-video feature CSVs (weak+strong mAP)
OUT_DIR     = ROOT / "weights" / "xgboost"

MODEL_WEAK   = "FASTERRCNN-MOBILENET-V3-LARGE-FPN"
MODEL_STRONG = "FASTERRCNN-RESNET50-FPN-V2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",  type=int, default=7,
                        help="Sliding-window size in frames (default: 7)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Prediction horizon in frames (default: 1)")
    parser.add_argument("--model-type", choices=["xgboost", "linear", "rf"],
                        default="xgboost")
    parser.add_argument("--derived-dir", type=str, default=None,
                        help="Override path to per-video feature CSVs")
    parser.add_argument("--out", type=str, default=None,
                        help="Output pkl path (default: weights/xgboost/predictor.pkl)")
    args = parser.parse_args()

    derived_dir = Path(args.derived_dir) if args.derived_dir else \
                  DERIVED_DIR / f"{MODEL_WEAK}_vs_{MODEL_STRONG}"

    if not derived_dir.exists():
        sys.exit(
            f"Derived feature directory not found: {derived_dir}\n"
            "You need per-video CSVs containing both weak-model features\n"
            "and the ∆mAP target.  These are produced by running weak and\n"
            "strong detection then merging with scoring/compute_xgboost.py --derive-only."
        )

    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_videos = splits["ua-detrac"]["train"]
    # Keep only videos whose feature CSV exists
    train_videos = [v for v in train_videos if (derived_dir / f"{v}.csv").exists()]
    if not train_videos:
        sys.exit("No feature CSVs found — check --derived-dir.")

    print(f"Training on {len(train_videos)} UA-DETRAC train videos  "
          f"(window={args.window}, horizon={args.horizon})")

    predictor = DifficultyPredictor(
        model_type=args.model_type,
        window_size=args.window,
        predict_horizon=args.horizon,
    )
    X_train, y_train = predictor.prepare_dataset(derived_dir, train_videos)
    print(f"Prepared {len(X_train)} training windows")
    predictor.train(X_train, y_train, verbose=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT_DIR / "predictor.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
