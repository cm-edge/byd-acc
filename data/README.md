# Data

This directory is not tracked by git. Download each dataset and place it here.

## Expected layout

```
data/
├── UA-DETRAC/
│   ├── DETRAC-Images/              # raw frames (JPEG, needed to train IoU-Net / GFL)
│   ├── DETRAC-Annos/
│   │   ├── DETRAC-Train-Annotations-XML/
│   │   └── DETRAC-Test-Annotations-XML/
│   └── DETRAC-Detections/
│       ├── DETRAC-Train-Detections/
│       │   └── FASTERRCNN-MOBILENET-V3-LARGE-FPN/   ← weak model outputs
│       └── DETRAC-Test-Detections/
│           ├── FASTERRCNN-MOBILENET-V3-LARGE-FPN/
│           └── FASTERRCNN-RESNET50-FPN-V2/          ← strong model outputs
│
├── UAVDT/
│   ├── UAV-benchmark-M/            # frames
│   └── UAV-benchmark-MOTD_v1.0/   # annotations
│
├── MOT17/
│   ├── train/
│   │   └── MOT17-{02,04,05,09,10,11,13}-FRCNN/
│   └── test/
│
├── mot17 -> MOT17                  # symlink, see note below
│
└── torch_hub/                      # cached torchvision model weights
```

**MOT17 path note:** some scripts use `data/MOT17` (the Jetson benchmark,
`compute_derived.py`, `compute_iounet_gfl.py`) and the MOT17 pipeline scripts
use lowercase `data/mot17` (`compute_mot17_*.py`, see
`dataset/README_mot17.md`). Both names must resolve. The simplest way is one
symlink after downloading:

```bash
cd data && ln -s MOT17 mot17
```

## Download links

| Dataset | URL |
|---|---|
| MOT17 | https://motchallenge.net/data/MOT17/ |
| UA-DETRAC | https://detrac-db.rit.albany.edu/ |
| UAVDT | https://sites.google.com/view/grli-uavdt/ |

Note: the original UA-DETRAC and UAVDT hosting pages have moved several times
since the datasets were released. If a link above is dead, search for the
dataset name plus "download"; both datasets are still mirrored by the
community. MOT17 downloads directly from motchallenge.net.

## Detection files format

Detection .txt files use the standard MOT-CSV format (one detection per line):
```
frame_id, obj_id, left, top, width, height, confidence, class_name
```

`obj_id` is a per-frame sequential id. Nothing downstream uses it (PTC
re-matches detections by IoU, XGBoost builds its own tracker), so its value
does not matter. See `detection/README.md`.

Detection files are regenerated in-repo by `detection/run_detection.py` (see
`detection/README.md`); MOT17 uses `scoring/compute_mot17_detections.py`. You do
not need to download precomputed detections — only the raw datasets above.
