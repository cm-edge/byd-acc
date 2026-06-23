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

### Step 3 — Train baselines (or use provided weights)
Pre-trained weights for IoU-Net, GFL, and XGBoost are in `weights/`.
To retrain from scratch:
```bash
python training/train_iounet.py      # ~4h on one GPU
python training/train_gfl.py        # ~4h on one GPU
python training/train_xgboost.py    # <10 min
```

### Step 4 — Compute per-frame scores
```bash
python scoring/compute_derived.py                        # per-frame AP, ~10 min
python scoring/compute_ptc.py                            # no GPU needed, ~30 min
python scoring/compute_iounet_gfl.py --method both      # GPU, ~2h
python scoring/compute_xgboost.py                       # CPU, ~15 min
```

### Step 5 — Reproduce figures and table
```bash
python reproduce/tab2.py    # prints Table 2
python reproduce/fig5.py    # → reproduce/output/fig5.pdf
python reproduce/fig6.py    # → reproduce/output/fig6.pdf
```

---

## Jetson-side results (fig2, fig3, fig4)

See `jetson/README.md` for Docker environment setup.

### Figure 2 (latency + throughput) — runs entirely inside container
```bash
# Inside Docker container:
python reproduce/jetson/fig2.py
# → reproduce/output/fig2.pdf
```

### Figure 3 (resources + energy) — needs two terminals
```bash
# Terminal 1 (HOST, outside Docker):
python reproduce/jetson/fig3.py --host-monitor

# Terminal 2 (Docker container):
python reproduce/jetson/fig3.py --container-bench

# After both finish:
python reproduce/jetson/fig3.py --plot-only
# → reproduce/output/fig3.pdf
```

### Figure 4 (motivation study) — same two-terminal pattern
```bash
# Terminal 1 (HOST):
python reproduce/jetson/fig4.py --host-monitor

# Terminal 2 (container):
python reproduce/jetson/fig4.py --container-bench

# Plot:
python reproduce/jetson/fig4.py --plot-only
# → reproduce/output/fig4.pdf
```
