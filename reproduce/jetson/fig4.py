#!/usr/bin/env python3
"""
Reproduce Appendix B Figure 4 — GPU/CPU utilisation motivation study.

Two-condition benchmark on Jetson Orin Nano:
  Condition A: Detector only (FasterRCNN-MNv3L-320)
  Condition B: Detector + GFL estimator (both on GPU)

Shows that co-locating the estimator on the GPU consumes the already-saturated
GPU budget while leaving the CPU idle — the key motivation for PTC-IoU.

HARDWARE REQUIREMENT: NVIDIA Jetson Orin Nano.

TWO-TERMINAL WORKFLOW (same as fig3.py):
  Terminal 1 (HOST):   python reproduce/jetson/fig4.py --host-monitor
  Terminal 2 (DOCKER): python reproduce/jetson/fig4.py --container-bench

Then plot:  python reproduce/jetson/fig4.py --plot-only

Output:
    reproduce/output/fig4.pdf
    reproduce/output/fig4.png
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

ROOT      = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "jetson" / "benchmark"
RESULTS   = BENCH_DIR / "results"
OUT_DIR   = Path(__file__).parent.parent / "output"

GPU_COLOR = "#4878CF"
CPU_COLOR = "#D65F5F"


def run_host_monitor():
    print("Starting host-side motivation study monitor...")
    print("In Terminal 2 (container), run:")
    print("  python reproduce/jetson/fig4.py --container-bench\n")
    subprocess.run([sys.executable, str(BENCH_DIR / "run_resources.py"),
                    "--mode", "motivation"], check=True)


def run_container_bench():
    """Run the two-condition benchmark (det-only vs det+GFL)."""
    print("Running two-condition motivation benchmark...")
    subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_latency.py"),
         "--conditions", "detector_only", "gfl_gpu"],
        check=True,
    )


def plot(out_dir: Path):
    res_file = RESULTS / "motivation_results.json"
    if not res_file.exists():
        sys.exit(
            f"Results not found: {res_file}\n"
            "Run --host-monitor and --container-bench first."
        )

    data = json.loads(res_file.read_text())
    det  = data.get("detector_only", {})
    est  = data.get("detector_plus_estimator_gpu", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Appendix B Fig 4: Motivation — GPU/CPU utilisation\n"
                 "FasterRCNN-MNv3L-320 on MOT17 (Jetson Orin Nano, batch 1)",
                 fontsize=9)

    x = np.array([0, 1])
    labels = ["Detector Only", "Detector +\nEstimator (GPU)"]
    w = 0.35

    # GPU utilisation
    gpu_vals = [det.get("gpu_util_mean", np.nan), est.get("gpu_util_mean", np.nan)]
    bars1 = ax1.bar(x, gpu_vals, w, color=GPU_COLOR, hatch="///",
                    edgecolor="black", linewidth=0.8, label="GPU Util (%)")
    for bar, v in zip(bars1, gpu_vals):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=9)

    ax1r = ax1.twinx()
    cpu_vals = [det.get("cpu_util_mean", np.nan), est.get("cpu_util_mean", np.nan)]
    bars2 = ax1r.bar(x + w, cpu_vals, w, color=CPU_COLOR, hatch="...",
                     edgecolor="black", linewidth=0.8, label="CPU Util (%)")
    for bar, v in zip(bars2, cpu_vals):
        if not np.isnan(v):
            ax1r.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                      f"{v:.1f}%", ha="center", fontsize=9)

    ax1.set_xticks(x + w/2)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("GPU Utilisation (%)", color=GPU_COLOR)
    ax1r.set_ylabel("CPU Utilisation (%)", color=CPU_COLOR)
    ax1.set_ylim(0, 110); ax1r.set_ylim(0, 30)
    ax1.legend(loc="upper left", fontsize=8)
    ax1r.legend(loc="upper right", fontsize=8)
    ax1.set_title("GPU/CPU Utilisation")

    # Latency (if available)
    lat_det = det.get("latency_mean_ms", np.nan)
    lat_est = est.get("latency_mean_ms", np.nan)
    if not (np.isnan(lat_det) or np.isnan(lat_est)):
        lat_vals = [lat_det, lat_est]
        ax2.bar(x + w/2, lat_vals, color=["#BBBBBB", GPU_COLOR],
                edgecolor="black", linewidth=0.8)
        for i, v in enumerate(lat_vals):
            ax2.text(x[i] + w/2, v + 0.5, f"{v:.1f} ms",
                     ha="center", fontsize=9)
        ax2.set_xticks(x + w/2)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel("Per-frame latency (ms)")
        ax2.set_title("Per-frame latency")
    else:
        ax2.text(0.5, 0.5, "Latency data\nnot available",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig4.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--host-monitor",    action="store_true")
    g.add_argument("--container-bench", action="store_true")
    g.add_argument("--plot-only",       action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    if args.host_monitor:
        run_host_monitor()
    elif args.container_bench:
        run_container_bench()
    elif args.plot_only:
        plot(Path(args.output_dir))
    else:
        print(__doc__)
        sys.exit(
            "Specify --host-monitor, --container-bench, or --plot-only."
        )


if __name__ == "__main__":
    main()
