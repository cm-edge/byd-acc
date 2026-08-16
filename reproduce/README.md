# Reproducing Paper Results

## Quick overview

| Script | Paper artifact | Hardware needed |
|---|---|---|
| `tab2.py` | Table 2: AP + pairwise accuracy @ ρ=0.3 | Any GPU server |
| `fig5.py` | Appendix C Fig 5: mAP vs offload curves | Any GPU server |
| `fig6.py` | Appendix C Fig 6: quality vs gain scatter | Any GPU server |
| `jetson/fig2.py` | Fig 2: latency + throughput | **Jetson Orin Nano** |
| `jetson/fig3.py` | Fig 3: resources + energy | **Jetson Orin Nano** |
| `jetson/fig4.py` | Appendix B Fig 4: GPU/CPU motivation | **Jetson Orin Nano** |

---

## Server-side results (tab2, fig5, fig6)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download datasets
See `data/README.md` for dataset download and layout.
For MOT17, `dataset/README_mot17.md` has the full walkthrough.

### Step 3 — Generate detections
```bash
python detection/run_detection.py --dataset ua-detrac
python detection/run_detection.py --dataset uavdt
python scoring/compute_mot17_detections.py --skip-existing
```

### Step 4 — Train baselines (or use provided weights)
Pre-trained weights for IoU-Net, GFL, and XGBoost are in `weights/`.
To retrain from scratch:
```bash
python training/train_iounet.py      # ~4h on one GPU
python training/train_gfl.py        # ~4h on one GPU
python training/train_xgboost.py    # <10 min
```

### Step 5 — Compute per-frame scores
```bash
python scoring/compute_derived.py                        # per-frame AP, ~10 min
python scoring/compute_ptc.py                            # no GPU needed, ~30 min
python scoring/compute_iounet_gfl.py --method both      # GPU, ~2h
python scoring/compute_xgboost.py                       # CPU, ~15 min
python scoring/compute_mot17_iounet_gfl.py --method both # MOT17 scores
python scoring/compute_mot17_routing.py                  # MOT17 routing curves
```

### Step 6 — Reproduce figures and table
```bash
python reproduce/tab2.py    # prints Table 2
python reproduce/fig5.py    # → reproduce/output/fig5.pdf
python reproduce/fig6.py    # → reproduce/output/fig6.pdf
```

---

## Jetson-side results (fig2, fig3, fig4)

The Jetson numbers depend on the machine state (power mode, CPU clocks,
background load). The bare figure scripts here do NOT set up that state, so
do not measure with them directly. Instead:

1. **Measure** by following [`REPRODUCING.md`](../REPRODUCING.md) at the repo
   root. It walks through host setup, the per-figure recipes (steps 5-8) and
   the expected tolerances. `jetson/README.md` covers the Docker environment.
2. **Plot** from the result JSONs afterwards:

```bash
python reproduce/jetson/fig2.py --plot-only
python reproduce/jetson/fig3.py --plot-only
python reproduce/jetson/fig4.py --plot-only
# → reproduce/output/fig{2,3,4}.pdf
```

The `--plot-only` commands read the result JSONs in
`jetson/benchmark/results/` (not tracked by git), so run the matching
REPRODUCING.md steps first.

`fig3.py` and `fig4.py` also have `--host-monitor` / `--container-bench`
wrapper modes for the two-terminal workflow. They run the underlying
benchmark and monitors for you, but they do not set up the background-load /
clock state either, so use them only after reading the matching step in
REPRODUCING.md (step 7 for fig3, step 8 for fig4). A monitored run overwrites
`latency.json`; save the step-6 copy first as REPRODUCING.md describes.

The directory layout of `reproduce/jetson/` (figure scripts and machine-state
tools) is documented in `reproduce/jetson/README.md`.
