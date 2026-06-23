#!/usr/bin/env python3
"""
Reproduce Figure 2 — Per-frame latency and sustained throughput on Jetson Orin Nano.

This script runs the full benchmark inside the Docker container, then plots
the results (Fig 2a: stacked latency bars; Fig 2b: throughput FPS bars).

HARDWARE REQUIREMENT: NVIDIA Jetson Orin Nano with JetPack 6.x.

DOCKER SETUP:
    See jetson/README.md for container build and mount instructions.
    This script must be run from inside the container:

        # Inside container:
        python reproduce/jetson/fig2.py

Two phases:
  1. Measurement  — runs jetson/benchmark/run_latency.py and
                    jetson/benchmark/run_throughput.py (may take ~10 min)
  2. Plotting     — reads results/ JSONs and writes fig2.pdf / fig2.png

Output:
    reproduce/output/fig2.pdf
    reproduce/output/fig2.png

Usage:
    python reproduce/jetson/fig2.py              # measure + plot
    python reproduce/jetson/fig2.py --plot-only  # re-plot from existing JSONs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
BENCH_DIR  = ROOT / "jetson" / "benchmark"
RESULTS    = BENCH_DIR / "results"
OUT_DIR    = Path(__file__).parent.parent / "output"

CONDITIONS = ["detector_only", "iounet_gpu", "gfl_gpu", "ptc_cpu", "xgboost_cpu"]
LABELS     = ["Detector\nonly", "+IoU-Net\n(GPU)", "+GFL\n(GPU)",
              "+PTC-IoU\n(CPU)", "+XGBoost\n(CPU)"]
COLORS     = ["#BBBBBB", "#4878CF", "#4878CF", "#4CAF50", "#D65F5F"]
HATCHES    = ["", "///", "...", "**", "xx"]


def run_benchmark():
    print("=" * 60)
    print("Phase 1: Running latency benchmark (run_latency.py)")
    print("=" * 60)
    subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_latency.py")],
        check=True,
    )

    print("\n" + "=" * 60)
    print("Phase 2: Running throughput benchmark (run_throughput.py)")
    print("=" * 60)
    subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_throughput.py")],
        check=True,
    )


def plot(out_dir: Path):
    lat_file  = RESULTS / "latency.json"
    tput_file = RESULTS / "throughput.json"

    if not lat_file.exists() or not tput_file.exists():
        sys.exit(
            f"Results not found in {RESULTS}/\n"
            "Run without --plot-only to measure first."
        )

    lat  = json.loads(lat_file.read_text())
    tput = json.loads(tput_file.read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fig 2: System performance on Jetson Orin Nano (MOT17, batch 1)",
                 fontsize=10)

    # ── (a) Latency ───────────────────────────────────────────────────────────
    det_mean = lat.get("detector_only", {}).get("mean_ms", 0)
    means, stds = [], []
    for cond in CONDITIONS:
        d = lat.get(cond, {})
        means.append(d.get("mean_ms", 0))
        stds.append(d.get("std_ms", 0))

    x = np.arange(len(CONDITIONS))
    bars = ax1.barh(x, means, xerr=stds, color=COLORS, hatch=HATCHES,
                    edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 0.8})
    ax1.axvline(det_mean, color="red", ls="--", lw=1, label="detector-only floor")
    for i, (m, s) in enumerate(zip(means, stds)):
        delta = m - det_mean
        sign  = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        ax1.text(m + s + 1, i, f"{m:.1f}±{s:.1f} ms  ({sign})",
                 va="center", fontsize=7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(LABELS, fontsize=8)
    ax1.set_xlabel("Per-frame latency (ms)")
    ax1.set_title("(a) Per-frame latency")
    ax1.legend(fontsize=8)

    # ── (b) Throughput ────────────────────────────────────────────────────────
    tput_conditions = [
        ("iounet_gpu", "IoU-Net\n(GPU)"),
        ("gfl_gpu",    "GFL\n(GPU)"),
        ("ptc_cpu_seq",      "PTC-IoU\nseq."),
        ("ptc_cpu_pipelined","PTC-IoU\npipe."),
        ("xgboost_cpu_seq",      "XGBoost\nseq."),
        ("xgboost_cpu_pipelined","XGBoost\npipe."),
    ]
    gpu_baseline = tput.get("iounet_gpu", {}).get("fps_mean", 0)
    fps_vals = [tput.get(k, {}).get("fps_mean", 0) for k, _ in tput_conditions]
    tc_labels = [l for _, l in tput_conditions]
    tc_colors = ["#4878CF", "#4878CF", "#4CAF50", "#4CAF50", "#D65F5F", "#D65F5F"]

    x2 = np.arange(len(tput_conditions))
    ax2.bar(x2, fps_vals, color=tc_colors, edgecolor="black", linewidth=0.8)
    ax2.axhline(gpu_baseline, color="red", ls="--", lw=1, label="GPU baseline")
    for i, v in enumerate(fps_vals):
        mult = v / gpu_baseline if gpu_baseline > 0 else 0
        ax2.text(i, v + 0.05, f"{v:.2f}\n({mult:.2f}×)",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(tc_labels, fontsize=8)
    ax2.set_ylabel("Throughput (FPS)")
    ax2.set_title("(b) Sustained throughput")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig2.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip measurement, plot from existing JSONs")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    if not args.plot_only:
        run_benchmark()

    plot(Path(args.output_dir))


if __name__ == "__main__":
    main()
