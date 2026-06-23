#!/usr/bin/env python3
"""Regenerate weak/strong detection .txt files for UA-DETRAC and UAVDT.

Self-contained replacement for the precomputed detection files: download the
public datasets into the expected layout (see adapters.py docstrings) and run
this to regenerate the per-video detection files the scoring scripts consume.

Output format (one line per detection, comma-separated):
    frame_id, obj_id, left, top, width, height, conf, class_name
  - frame_id : 1-based, in frame-sort order.
  - obj_id   : per-frame sequential id (1..N). NOTE: nothing downstream uses this
               column — PTC re-matches by IoU and XGBoost rebuilds its own
               internal IoU tracker — so its exact value is irrelevant.
  - box      : pixel coords, left/top/width/height.

Pinned reproduction settings live in config.py (model variants, conf=0.0,
per-dataset class filter). MOT17 is generated separately by
scoring/compute_mot17_detections.py.

Usage:
    python detection/run_detection.py --dataset ua-detrac
    python detection/run_detection.py --dataset uavdt --models weak strong
    python detection/run_detection.py --dataset ua-detrac --split train
    python detection/run_detection.py --dataset ua-detrac --skip-existing
    python detection/run_detection.py --dataset uavdt --data-root /data/UAVDT
"""

import argparse
import sys
from pathlib import Path

# Make sibling modules importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from adapters import DATASET_NAMES, DEFAULT_DATA_ROOT, get_adapter
from config import (
    DATASET_CLASS_FILTER,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONF,
    DEFAULT_MODELS,
    MODEL_REGISTRY,
)
from detector import Detector

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_video_frames(frame_dir: Path, glob_pattern: str):
    return sorted(frame_dir.glob(glob_pattern))


def process_video(frame_dir, detector, output_file, batch_size, frame_glob):
    """Detect all frames of one video and write its detection .txt."""
    frames = get_video_frames(frame_dir, frame_glob)
    if not frames:
        print(f"  [WARN] no frames in {frame_dir}")
        return 0

    lines = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_results = detector.detect_batch(batch)
        for batch_idx, frame_dets in enumerate(batch_results):
            frame_id = i + batch_idx + 1  # 1-based
            for obj_id, (left, top, w, h, conf, cls) in enumerate(frame_dets, start=1):
                lines.append(
                    f"{frame_id},{obj_id},{left:.2f},{top:.2f},"
                    f"{w:.2f},{h:.2f},{conf:.6f},{cls}\n"
                )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(lines))
    return len(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate UA-DETRAC / UAVDT detection files in-repo")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root (default: <repo>/data/{DATASET})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODEL_REGISTRY),
                        help="Models to run (default: weak strong)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="Confidence threshold (default: 0.0, save all)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="Split to process (UA-DETRAC; UAVDT is single-split)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video names to process (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose detection .txt already exists")
    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        parser.error(f"--conf must be in [0,1], got {args.conf}")

    # Resolve data root.
    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        data_root = REPO_ROOT / DEFAULT_DATA_ROOT[args.dataset]

    adapter = get_adapter(args.dataset)
    class_filter = DATASET_CLASS_FILTER[args.dataset]

    # Build video list.
    if args.videos:
        video_names = args.videos
    elif args.dataset == "ua-detrac" and args.split != "all":
        video_names = adapter.list_videos(data_root, split=args.split)
    else:
        video_names = adapter.list_videos(data_root)

    if not video_names:
        raise SystemExit(
            f"No videos found under {data_root}. Check the dataset layout "
            f"(see detection/adapters.py docstrings)."
        )

    frame_glob = adapter.get_frame_glob()

    print(f"\n{'='*70}")
    print(f"Dataset:       {args.dataset}")
    print(f"Data root:     {data_root}")
    print(f"Videos:        {len(video_names)}")
    print(f"Models:        {', '.join(args.models)}")
    print(f"Conf:          {args.conf}")
    print(f"Class filter:  {sorted(class_filter)} (1-based COCO ids)")
    print(f"Batch size:    {args.batch_size}")
    print(f"{'='*70}\n")

    for model_key in args.models:
        model_folder = MODEL_REGISTRY[model_key]
        print(f"\n{'='*70}\nLoading model: {model_key} -> {model_folder}\n{'='*70}")
        detector = Detector(model_key, args.conf, class_filter=class_filter)

        skipped = 0
        for video_name in tqdm(video_names, desc=model_folder):
            output_file = adapter.get_output_file(model_folder, video_name, data_root)
            if args.skip_existing and output_file.exists():
                skipped += 1
                continue
            frame_dir = adapter.get_frame_dir(video_name, data_root)
            if not frame_dir.exists():
                print(f"  [WARN] frame dir not found: {frame_dir}, skipping {video_name}")
                continue
            process_video(frame_dir, detector, output_file,
                          batch_size=args.batch_size, frame_glob=frame_glob)

        if skipped:
            print(f"  Skipped {skipped} existing videos for {model_folder}")

    print("\nDone.")


if __name__ == "__main__":
    main()
