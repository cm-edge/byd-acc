#!/usr/bin/env python3
"""
Reproduce Figure 3 — Resource utilisation and energy footprint on Jetson Orin Nano.

HARDWARE REQUIREMENT: NVIDIA Jetson Orin Nano with JetPack 6.x.
ENERGY MEASUREMENT requires jetson-stats (jtop) installed on the HOST.

This figure needs two terminals running concurrently:

  Terminal 1 (HOST — outside Docker):
      python reproduce/jetson/fig3.py --host-monitor
      # Starts tegrastats + jtop collection, waits for the container.

  Terminal 2 (DOCKER CONTAINER):
      python reproduce/jetson/fig3.py --container-bench
      # Runs the latency benchmark, writes timing markers.

After both finish, plot with:
      python reproduce/jetson/fig3.py --plot-only

For the machine state used by the reference runs (background load, taskset,
LAZY_XGB_IMPORT=1), follow REPRODUCING.md step 7 instead of the bare
--container-bench command.

Output:
    reproduce/output/fig3.pdf
    reproduce/output/fig3.png
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

CONDITIONS = ["detector_only", "iounet_gpu", "gfl_gpu", "ptc_cpu", "xgboost_cpu"]
LABELS     = ["Det.\nonly", "+IoU-Net\n(GPU)", "+GFL\n(GPU)",
              "+PTC-IoU\n(CPU)", "+XGBoost\n(CPU)"]
GPU_COLOR  = "#4878CF"
CPU_COLOR  = "#D65F5F"
PWR_COLOR  = "#E74C3C"
ENE_COLOR  = "#F39C12"


def run_host_monitor():
    print("Starting host-side resource + energy monitor...")
    print("Switch to the container terminal and run:")
    print("  python reproduce/jetson/fig3.py --container-bench\n")
    # Both monitors wait for the timing markers the container benchmark writes,
    # so they must run at the same time, not one after the other.
    p_res = subprocess.Popen([sys.executable, str(BENCH_DIR / "run_resources.py")])
    p_eng = subprocess.Popen([sys.executable, str(BENCH_DIR / "run_energy.py")])
    rc_res = p_res.wait()
    rc_eng = p_eng.wait()
    if rc_res != 0 or rc_eng != 0:
        sys.exit(f"Monitor failed (run_resources rc={rc_res}, run_energy rc={rc_eng})")


def run_container_bench():
    print("Running latency benchmark (writes timing markers for host monitor)...")
    subprocess.run(
        [sys.executable, str(BENCH_DIR / "run_latency.py")],
        check=True,
    )


def plot(out_dir: Path):
    res_file = RESULTS / "resources.json"
    eng_file = RESULTS / "energy.json"

    missing = [str(f) for f in (res_file, eng_file) if not f.exists()]
    if missing:
        sys.exit(
            f"Missing result files: {missing}\n"
            "Run --host-monitor and --container-bench first."
        )

    res = json.loads(res_file.read_text())
    eng = json.loads(eng_file.read_text())

    # Fig 3(b) in the paper is measured on the THROUGHPUT run (sequential and
    # pipelined variants), not the latency run, so it has its own source file.
    thr_file = RESULTS / "resources_throughput.json"
    res_thr = json.loads(thr_file.read_text()) if thr_file.exists() else {}

    x = np.arange(len(CONDITIONS))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Fig 3: Resource and energy footprint on Jetson Orin Nano", fontsize=10)

    # ── (a) CPU util Δ + RAM Δ (relative to detector-only, as in the paper) ────
    ax = axes[0]
    base = res.get("detector_only", {})
    base_cpu = base.get("cpu_pct_mean_mean", np.nan)
    base_ram = base.get("ram_used_mb_mean", np.nan)
    est_conds = [c for c in CONDITIONS if c != "detector_only"]
    xe = np.arange(len(est_conds))
    cpu_d = [res.get(c, {}).get("cpu_pct_mean_mean", np.nan) - base_cpu for c in est_conds]
    ram_d = [res.get(c, {}).get("ram_used_mb_mean", np.nan) - base_ram for c in est_conds]
    ax.bar(xe, cpu_d, color=CPU_COLOR, alpha=0.7, label="CPU util Δ (%)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax2 = ax.twinx()
    ax2.plot(xe, ram_d, "ko--", ms=5, lw=1.2, label="RAM overhead (MB)")
    ax.set_xticks(xe); ax.set_xticklabels(LABELS[1:], fontsize=8)
    ax.set_ylabel("CPU Util Change vs Det-Only (%)", color=CPU_COLOR)
    ax2.set_ylabel("RAM overhead (MB)")
    ax.set_title("(a) CPU Util. & RAM Overhead")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # ── (b) GPU util (throughput run: seq + pipelined) ────────────────────────
    ax = axes[1]
    thr_conds = ["iounet_gpu", "gfl_gpu", "ptc_cpu_seq", "ptc_cpu_pipelined",
                 "xgboost_cpu_seq", "xgboost_cpu_pipelined"]
    thr_labels = ["IoUNet\n(GPU)", "GFL\n(GPU)", "PTC\n(seq)", "PTC\n(pipe)",
                  "XGB\n(seq)", "XGB\n(pipe)"]
    src = res_thr if res_thr else res
    gpu_vals = [src.get(c, {}).get("gpu_pct_mean", np.nan) for c in thr_conds]
    xg = np.arange(len(thr_conds))
    ax.bar(xg, gpu_vals, color=GPU_COLOR, alpha=0.8)
    for i, v in enumerate(gpu_vals):
        if not np.isnan(v):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_xticks(xg); ax.set_xticklabels(thr_labels, fontsize=8)
    ax.set_ylabel("GPU Utilisation (%)")
    ax.set_title("(b) GPU Utilization")

    # ── (c) Power + energy overhead ───────────────────────────────────────────
    ax = axes[2]
    pwr_vals = [eng.get(c, {}).get("power_mean_mw", np.nan) for c in CONDITIONS]
    # Energy overhead is the per-frame energy (power x time) change vs detector-only,
    # already computed by run_energy.py. Use it directly rather than re-deriving from
    # instantaneous power, which moves in the opposite direction (a head can draw more
    # power yet less energy because it finishes the frame sooner).
    ene_pct  = [eng.get(c, {}).get("energy_overhead_pct", np.nan) for c in CONDITIONS]

    ax.bar(x, pwr_vals, color=PWR_COLOR, alpha=0.7, label="Power (mW)")
    ax3 = ax.twinx()
    ax3.plot(x, ene_pct, "s--", color=ENE_COLOR, ms=5, lw=1.2,
             label="Energy overhead (%)")
    ax3.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("Average Power (mW)", color=PWR_COLOR)
    ax3.set_ylabel("Energy Overhead vs Det-Only (%)", color=ENE_COLOR)
    ax.set_title("(c) Power & Energy Overhead")
    ax.legend(loc="upper left", fontsize=7)
    ax3.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig3.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--host-monitor",    action="store_true",
                   help="Run resource + energy collection on the HOST")
    g.add_argument("--container-bench", action="store_true",
                   help="Run latency benchmark inside the Docker container")
    g.add_argument("--plot-only",       action="store_true",
                   help="Plot from existing result JSONs (no measurement)")
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
            "Please specify one of:\n"
            "  --host-monitor     (run on HOST in Terminal 1)\n"
            "  --container-bench  (run in Docker container in Terminal 2)\n"
            "  --plot-only        (after both complete)"
        )


if __name__ == "__main__":
    main()
