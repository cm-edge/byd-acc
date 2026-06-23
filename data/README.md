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
└── torch_hub/                      # cached torchvision model weights
```

## Download links

| Dataset | URL |
|---|---|
| UA-DETRAC | https://detrac-db.rit.albany.edu/ |
| UAVDT | https://sites.google.com/site/daviddo87/projects/uavdt |
| MOT17 | https://motchallenge.net/data/MOT17/ |

## Detection files format

Detection .txt files use the standard MOT-CSV format (one detection per line):
```
frame_id, track_id, left, top, width, height, confidence, class_name
```

Detection files are regenerated in-repo by `detection/run_detection.py` (see
`detection/README.md`); MOT17 uses `scoring/compute_mot17_detections.py`. You do
not need to download precomputed detections — only the raw datasets above.
