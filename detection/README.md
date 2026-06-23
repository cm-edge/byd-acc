# Detection module (UA-DETRAC, UAVDT)

Self-contained regeneration of the weak/strong detection files the scoring
scripts consume. Instead of committing the (large) precomputed detection
`.txt` files, download the public datasets into the expected layout and run the
detector here.

**MOT17 is handled separately** by `scoring/compute_mot17_detections.py` — this
module only covers UA-DETRAC and UAVDT.

## Models

| key      | model                                | role   | output folder name                     |
|----------|--------------------------------------|--------|----------------------------------------|
| `weak`   | `fasterrcnn_mobilenet_v3_large_fpn`  | weak   | `FASTERRCNN-MOBILENET-V3-LARGE-FPN`    |
| `strong` | `fasterrcnn_resnet50_fpn_v2`         | strong | `FASTERRCNN-RESNET50-FPN-V2`           |

COCO-pretrained weights auto-download on first run. The weak model is the
**full-resolution** MobileNet FPN (min 800 / max 1333), *not* the literal
`_320_fpn` variant — the 320 model downscales frames and badly weakens
detections (same trap documented in `scoring/compute_mot17_detections.py`).

## Pinned reproduction settings (`config.py`)

- `transform = weights.transforms()` — no manual resize.
- `conf = 0.0` — save all (torchvision floors at `box_score_thresh ≈ 0.05`);
  downstream scripts apply their own thresholds.
- Per-dataset class filter (1-based COCO ids):
  - UA-DETRAC: `{3,4,6,8}` = car / motorcycle / bus / truck
  - UAVDT: `{3,6,8}` = car / bus / truck (no motorcycle)

## Output format

One line per detection: `frame_id, obj_id, left, top, width, height, conf, class_name`

- `frame_id`: 1-based, frame-sort order.
- `obj_id`: per-frame sequential id. **Nothing downstream uses this column** —
  PTC re-matches detections by IoU and XGBoost rebuilds its own internal IoU
  tracker — so its exact value is irrelevant. No external tracker is involved.

## Expected dataset layout

```
data/UA-DETRAC/
    DETRAC-Images/MVI_xxxxx/img00001.jpg ...
    DETRAC-Detections/DETRAC-{Train,Test}-Detections/{MODEL}/   (written)

data/UAVDT/
    UAV-benchmark-M/M0101/img000001.jpg ...
    Detections/{MODEL}/                                          (written)
```

## Usage

```bash
python detection/run_detection.py --dataset ua-detrac                 # weak+strong
python detection/run_detection.py --dataset uavdt --models weak strong
python detection/run_detection.py --dataset ua-detrac --split train
python detection/run_detection.py --dataset ua-detrac --skip-existing
python detection/run_detection.py --dataset uavdt --data-root /path/to/UAVDT
```

## Reproduction fidelity

Regenerated detections are **not byte-identical** to the reference files (GPU
non-determinism + torchvision version), and the reference files on disk also
carry a stale generation artifact (a low-confidence drop after frame 3 that the
reference code itself no longer reproduces). High-confidence detections — the
ones that drive AP and the PTC/XGBoost signals — match the reference per frame.
The target is reproducing the paper's AP / ranking within noise, not per-box
identity.
