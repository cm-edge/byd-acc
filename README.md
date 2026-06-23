# Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection

Code for the paper **"Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection"** (Paper #53).

---

## Overview

Online object detectors at the edge need a per-frame signal of how trustworthy their predictions are, with no access to ground truth.  We propose **PTC-IoU** — a deterministic, non-learned estimator built from temporal consistency of the detector's output stream.  It matches two established learned baselines (IoU-Net, GFL) on estimation quality while adding only **0.6 ms per frame** (≈ 20× less than a co-located GPU head).

---

## Repository layout

```
├── estimator/          Core library: PTC-IoU, IoU-Net, GFL, XGBoost (server + Jetson)
├── detection/          Regenerate weak/strong detections in-repo (UA-DETRAC, UAVDT)
├── training/           Train baselines from scratch (requires raw video frames + GPU)
├── scoring/            Compute per-frame estimator scores from detections
├── dataset/            Dataset adapters and video-split definitions
├── weights/            Pre-trained model weights (IoU-Net, GFL, XGBoost)
├── jetson/             Jetson Orin Nano benchmark (latency / throughput / energy)
│   └── benchmark/
├── reproduce/          One script per paper figure / table
│   ├── tab2.py         → Table 2: quality @ ρ=0.3
│   ├── fig5.py         → Appendix C Fig 5: mAP vs offload curves
│   ├── fig6.py         → Appendix C Fig 6: quality vs routing gain
│   └── jetson/
│       ├── fig2.py     → Fig 2: latency + throughput  [Jetson]
│       ├── fig3.py     → Fig 3: resources + energy    [Jetson]
│       └── fig4.py     → Appendix B Fig 4: motivation [Jetson]
└── data/               Datasets (not tracked — see data/README.md)
```

---

## Quickstart — server-side results

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download datasets → see data/README.md

# 3. Generate detections (UA-DETRAC, UAVDT → see detection/README.md;
#    MOT17 uses a separate script)
python detection/run_detection.py --dataset ua-detrac
python detection/run_detection.py --dataset uavdt
python scoring/compute_mot17_detections.py

# 4. Compute scores (pre-trained weights already in weights/)
python scoring/compute_ptc.py
python scoring/compute_iounet_gfl.py --method both
python scoring/compute_xgboost.py

# 5. Reproduce
python reproduce/tab2.py          # Table 2
python reproduce/fig5.py          # Appendix C Fig 5
python reproduce/fig6.py          # Appendix C Fig 6
```

## Quickstart — Jetson results

Requires NVIDIA Jetson Orin Nano.  See `jetson/README.md` for Docker setup.

```bash
# Fig 2 (latency + throughput) — inside container:
python reproduce/jetson/fig2.py

# Fig 3 (resources + energy) — two terminals:
# Terminal 1 (host):      python reproduce/jetson/fig3.py --host-monitor
# Terminal 2 (container): python reproduce/jetson/fig3.py --container-bench
# Then:                   python reproduce/jetson/fig3.py --plot-only
```

Full instructions: `reproduce/README.md`

---

## Estimators

| Estimator | Type | GPU needed | Parameters |
|---|---|---|---|
| **PTC-IoU** | Temporal consistency (Det × Ass × Loc) | No | 0 |
| **XGBoost** | Learned, CPU-resident | No | ~12 M (tree ensemble) |
| IoU-Net | Learned, GPU-resident | Yes | ~12.6 M |
| GFL | Learned, GPU-resident | Yes | ~12.6 M |

---

## Citation

```bibtex
@inproceedings{ptciou2026,
  title     = {Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection},
  booktitle = {TODO},
  year      = {2026},
}
```
