# Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection

Code and reproduction package for the paper **"Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection"** (Paper #53).

---

## Overview

An object detector deployed at the edge needs a per-frame signal of how much its predictions can be trusted, but it has no ground truth to check against. We propose **PTC-IoU**, a deterministic, non-learned estimator built from the temporal consistency of the detector's own output stream. It matches two learned baselines (IoU-Net, GFL) on estimation quality, while its cost is closed-form box arithmetic that runs on the otherwise idle CPU instead of the saturated GPU.

---

## Repository layout

```
├── REPRODUCING.md      Step-by-step manual for the Jetson results (start here for Fig 2/3/4)
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

## Quickstart — server-side results (Table 2, Fig 5, Fig 6)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download datasets
#    See data/README.md for layout and links.
#    For the MOT17 part, dataset/README_mot17.md has the full walkthrough.

# 3. Generate detections
python detection/run_detection.py --dataset ua-detrac
python detection/run_detection.py --dataset uavdt
python scoring/compute_mot17_detections.py --skip-existing

# 4. Compute per-frame scores (pre-trained weights already in weights/)
python scoring/compute_derived.py
python scoring/compute_ptc.py
python scoring/compute_iounet_gfl.py --method both
python scoring/compute_xgboost.py
python scoring/compute_mot17_iounet_gfl.py --method both
python scoring/compute_mot17_routing.py

# 5. Reproduce
python reproduce/tab2.py          # Table 2
python reproduce/fig5.py          # Appendix C Fig 5
python reproduce/fig6.py          # Appendix C Fig 6
```

## Quickstart — Jetson results (Fig 2, Fig 3, Fig 4)

Requires an NVIDIA Jetson Orin Nano. **Follow [`REPRODUCING.md`](REPRODUCING.md) step by step.** The system-performance numbers depend on the machine state (power mode, CPU clocks, background load), and that manual pins the state down and gives the expected value and tolerance for every quantity. `jetson/README.md` documents the Docker environment and the individual benchmark scripts.

The benchmark runs write their result JSONs to `jetson/benchmark/results/`
(not tracked by git). After measuring, plot the figures with:

```bash
python reproduce/jetson/fig2.py --plot-only
python reproduce/jetson/fig3.py --plot-only
python reproduce/jetson/fig4.py --plot-only
```

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
@misc{ptciou2026,
  title = {Beyond Accuracy: Rethinking Resource-Efficient Correctness Estimation in Edge Detection},
  year  = {2026},
  note  = {under review},
}
```
