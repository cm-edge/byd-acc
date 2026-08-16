#!/usr/bin/env python3
"""
Run TorchVision detectors on MOT17 training sequences and save one XML per frame.

Uses COCO pretrained weights (auto-downloaded on first run, no local checkpoint needed):
  - Weak:   fasterrcnn_mobilenet_v3_large_320_fpn  (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
  - Strong: fasterrcnn_resnet50_fpn_v2             (FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

All detections are saved (conf >= 0.0). Downstream scripts apply their own threshold.

Dataset layout expected (set --mot17-root to override):
    <mot17-root>/
        MOT17-02-FRCNN/
            img1/
                000001.jpg
                000002.jpg
                ...
            gt/
                gt.txt
        MOT17-04-FRCNN/
            ...
        ...   (all 7 FRCNN train sequences)

Output:
    data/mot17/pred_xml/<split>/<model_name>/<seq>/000001.xml ...

XML format (consumed by compute_mot17_routing.py and compute_derived.py):
    <annotation>
      <filename>000001.jpg</filename>
      <object>
        <name>person</name>
        <confidence>0.923456</confidence>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>80</xmax><ymax>160</ymax></bndbox>
      </object>
    </annotation>

Usage:
    python scoring/compute_mot17_detections.py
    python scoring/compute_mot17_detections.py --mot17-root /data/MOT17/train
    python scoring/compute_mot17_detections.py --models weak          # weak only
    python scoring/compute_mot17_detections.py --models strong        # strong only
    python scoring/compute_mot17_detections.py --models weak strong   # both (default)
    python scoring/compute_mot17_detections.py --skip-existing        # resume interrupted run
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_MOT17_ROOT = ROOT / "data" / "mot17" / "train"
XML_OUT_BASE       = ROOT / "data" / "mot17" / "pred_xml"

# NOTE: the weak model's output folder is named "...320_fpn" for historical
# compatibility (downstream scripts expect that folder name), but the actual
# model is the FULL-resolution fasterrcnn_mobilenet_v3_large_fpn (min_size=800,
# max_size=1333). Using the literal _320_fpn model (min 320/max 640) downscales
# 1920x1080 frames and badly weakens detections (weak AP 0.21 vs 0.34).
MODEL_REGISTRY = {
    "weak":   ("fasterrcnn_mobilenet_v3_large_320_fpn",
               fasterrcnn_mobilenet_v3_large_fpn,
               FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "strong": ("fasterrcnn_resnet50_fpn_v2",
               fasterrcnn_resnet50_fpn_v2,
               FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}

# Only process FRCNN sequences (avoids running on DPM/SDP which share frames)
FRCNN_SUFFIX = "FRCNN"

BATCH_SIZE = 8
PERSON_LABEL = 1  # COCO class id for person


# ---------------------------------------------------------------------------
# XML writer
# ---------------------------------------------------------------------------
def _save_xml(xml_path: Path, filename: str,
              dets: List[Tuple[float, float, float, float, float]]):
    """dets: list of (xmin, ymin, xmax, ymax, conf) in pixel coords."""
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = filename

    for xmin, ymin, xmax, ymax, conf in dets:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = "person"
        ET.SubElement(obj, "confidence").text = f"{conf:.6f}"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(max(0, int(round(xmin))))
        ET.SubElement(bb, "ymin").text = str(max(0, int(round(ymin))))
        ET.SubElement(bb, "xmax").text = str(max(0, int(round(xmax))))
        ET.SubElement(bb, "ymax").text = str(max(0, int(round(ymax))))

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ann).write(str(xml_path), encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------
def run_detector_on_seq(model, transform, seq_img_dir: Path,
                        xml_out_dir: Path, skip_existing: bool,
                        batch_size: int = BATCH_SIZE):
    img_paths = sorted(seq_img_dir.glob("*.jpg")) + sorted(seq_img_dir.glob("*.png"))
    if not img_paths:
        print(f"  [WARN] no images in {seq_img_dir}")
        return 0

    xml_out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    total_frames = 0
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]

        if skip_existing:
            missing = [p for p in batch_paths
                       if not (xml_out_dir / (p.stem + ".xml")).exists()]
            if not missing:
                total_frames += len(batch_paths)
                continue
            batch_paths = missing

        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            preds = model(batch)

        for img_path, pred in zip(batch_paths, preds):
            boxes  = pred["boxes"].cpu()
            labels = pred["labels"].cpu()
            scores = pred["scores"].cpu()

            dets = []
            for box, label, score in zip(boxes, labels, scores):
                if int(label) != PERSON_LABEL:
                    continue
                x1, y1, x2, y2 = box.tolist()
                dets.append((x1, y1, x2, y2, float(score)))

            xml_path = xml_out_dir / (img_path.stem + ".xml")
            _save_xml(xml_path, img_path.name, dets)
            total_frames += 1

    return total_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-frame XML detections for MOT17")
    parser.add_argument("--mot17-root", type=str, default=str(DEFAULT_MOT17_ROOT),
                        help="Path to MOT17 train directory (contains MOT17-*-FRCNN/)")
    parser.add_argument("--out-dir", type=str, default=str(XML_OUT_BASE),
                        help="Output base dir (default: data/mot17/pred_xml/)")
    parser.add_argument("--split", type=str, default="TRAIN",
                        help="Split label used in output path (default: TRAIN)")
    parser.add_argument("--models", nargs="+", choices=["weak", "strong"],
                        default=["weak", "strong"])
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose XML already exists (resume)")
    parser.add_argument("--seqs", nargs="+", default=None,
                        help="Process only these sequences (e.g. MOT17-02-FRCNN)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Frames per forward pass (default 8; use 1-2 on "
                             "small-memory devices like Jetson)")
    args = parser.parse_args()

    mot17_root = Path(args.mot17_root)
    out_base   = Path(args.out_dir)

    if not mot17_root.is_dir():
        raise SystemExit(
            f"MOT17 root not found: {mot17_root}\n"
            f"Set --mot17-root to the directory containing MOT17-*-FRCNN/ folders."
        )

    # Discover sequences
    all_seqs = sorted(
        d for d in mot17_root.iterdir()
        if d.is_dir() and d.name.endswith(f"-{FRCNN_SUFFIX}") and (d / "img1").is_dir()
    )
    if args.seqs:
        keep = set(args.seqs)
        all_seqs = [d for d in all_seqs if d.name in keep]
    if not all_seqs:
        raise SystemExit(f"No FRCNN sequences found under {mot17_root}")

    print(f"Sequences: {[d.name for d in all_seqs]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for model_key in args.models:
        model_name, model_fn, weights = MODEL_REGISTRY[model_key]
        print(f"\n{'='*70}")
        print(f"Model: {model_name}  [{model_key}]")
        print(f"{'='*70}")

        print("  Loading weights (auto-download if needed)...")
        model = model_fn(weights=weights).to(device)
        model.eval()
        transform = weights.transforms()

        for seq_dir in all_seqs:
            xml_out_dir = out_base / args.split / model_name / seq_dir.name
            already_done = xml_out_dir.exists() and any(xml_out_dir.glob("*.xml"))
            if already_done and args.skip_existing:
                n_xml = sum(1 for _ in xml_out_dir.glob("*.xml"))
                print(f"  [SKIP] {seq_dir.name}: {n_xml} XMLs already present")
                continue

            print(f"\n  Seq: {seq_dir.name}")
            n = run_detector_on_seq(
                model, transform,
                seq_img_dir=seq_dir / "img1",
                xml_out_dir=xml_out_dir,
                skip_existing=args.skip_existing,
                batch_size=args.batch_size,
            )
            print(f"    Done: {n} frames → {xml_out_dir}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
