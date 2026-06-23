#!/usr/bin/env python3
"""
Plot Tasks 1–4 experimental results (detector-only baseline visible).

  Task 1 (latency.json):               Stacked bars: detector base + estimator overhead
  Task 2 (throughput.json):            FPS bars coloured GPU/CPU + detector-only ref line
  Task 3 (resources_throughput.json):  3-panel vertical: GPU% / CPU% / RAM
  Task 4 (energy.json):                Power draw (left) + energy overhead % vs baseline (right)

Usage:
    python3 plot_tasks.py               # all individual PDFs + tasks_all.pdf
    python3 plot_tasks.py --task 1      # single task PDF
    python3 plot_tasks.py --combined    # combined 2x2 PDF only
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# ── Palette ───────────────────────────────────────────────────────────────────
DET_COLOR = "#BBBBBB"   # detector-only base (light gray)
GPU_COLOR = "#4878CF"   # GPU estimator (steel blue)
PTC_COLOR = "#4CAF50"   # PTC CPU estimator (green)
XGB_COLOR = "#D65F5F"   # XGBoost CPU estimator (brick red)
CPU_COLOR = "#D65F5F"   # CPU-util axis
RAM_COLOR = "#9B59B6"   # RAM axis (purple)
PWR_COLOR = "#E74C3C"   # power draw (red)
ENE_COLOR = "#F39C12"   # energy overhead (amber)

DET_HATCH = ""
GPU_HATCH = "///"
PTC_HATCH = "**"
XGB_HATCH = "..."
CPU_HATCH = "..."
RAM_HATCH = "xx"
PWR_HATCH = "+++"
ENE_HATCH = "oo"

EDGE_COLOR = "black"
EDGE_WIDTH = 0.8
BAR_W      = 0.50    # single bar per x-tick
BAR_W2     = 0.32    # two bars per x-tick (dual-axis)
FONTSIZE   = 13
LABEL_FS   = 12
HATCH_LW   = 0.5

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       FONTSIZE,
    "axes.labelsize":  FONTSIZE,
    "axes.titlesize":  FONTSIZE,
    "xtick.labelsize": LABEL_FS,
    "ytick.labelsize": LABEL_FS,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "hatch.linewidth": HATCH_LW,
})

# ── Conditions ────────────────────────────────────────────────────────────────
LAT_LABELS = ["Detector\nOnly", "IoUNet\n(GPU)", "GFL\n(GPU)", "PTC\n(CPU)", "XGBoost\n(CPU)"]
LAT_KEYS   = ["detector_only",  "iounet_gpu",    "gfl_gpu",    "ptc_cpu",    "xgboost_cpu"]
# colour/hatch for each estimator's overhead bar
LAT_OVH_COLOR = [DET_COLOR, GPU_COLOR, GPU_COLOR, PTC_COLOR, XGB_COLOR]
LAT_OVH_HATCH = [DET_HATCH, GPU_HATCH, GPU_HATCH, PTC_HATCH, XGB_HATCH]

THR_LABELS  = ["IoUNet\n(GPU)", "GFL\n(GPU)", "PTC\n(seq)", "PTC\n(pipe)", "XGB\n(seq)", "XGB\n(pipe)"]
THR_KEYS    = ["iounet_gpu", "gfl_gpu", "ptc_cpu_seq", "ptc_cpu_pipelined",
               "xgboost_cpu_seq", "xgboost_cpu_pipelined"]
THR_IS_CPU  = [False, False, True, True, True, True]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(fname: str) -> dict:
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)


def _bar(ax, x, means, stds, color, hatch, width=BAR_W, offset=0.0, bottom=0):
    """Draw a bar group; bottom can be a scalar for stacked bars."""
    return ax.bar(
        np.asarray(x, dtype=float) + offset, means, width=width,
        bottom=bottom,
        color=color, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, hatch=hatch,
        yerr=stds, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"},
        zorder=3,
    )


def _lbl(ax, bar, v, fmt="{:.1f}", dy=None):
    """Label centred just above a bar; dy=None → 3 % of total bar height."""
    top = bar.get_y() + bar.get_height()
    y   = top * 1.03 if dy is None else top + dy
    ax.text(
        bar.get_x() + bar.get_width() / 2, y,
        fmt.format(v),
        ha="center", va="bottom",
        fontsize=LABEL_FS - 1, fontweight="bold",
    )


def _style_l(ax, color):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)


def _style_r(ax, color):
    ax.spines[["top", "left"]].set_visible(False)
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)


def _ylim(ax, vals, stds=None, pad=1.5, hi=None, lo=0):
    stds = stds or [0] * len(vals)
    mx   = max(v + s for v, s in zip(vals, stds))
    up   = min(mx * pad, hi) if hi is not None else mx * pad
    ax.set_ylim(lo, up)


def _patch(color, hatch, label):
    return mpatches.Patch(facecolor=color, edgecolor=EDGE_COLOR,
                          linewidth=EDGE_WIDTH, hatch=hatch, label=label)


def _legend(ax, handles, ncol=None):
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol or len(handles), fontsize=LABEL_FS,
              framealpha=0.9, edgecolor="#cccccc")


def _refline(ax, y, label, color="dimgray"):
    ax.axhline(y, color=color, linewidth=1.0, linestyle="--", zorder=2, alpha=0.8)
    ax.text(0.01, y, f"  {label}", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=LABEL_FS - 1,
            color=color, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Stacked latency — gray base (detector) + coloured overhead per estimator
# ─────────────────────────────────────────────────────────────────────────────
def plot_task1(ax, data):
    base  = data["detector_only"]["mean_ms"]
    x     = np.arange(len(LAT_KEYS), dtype=float)
    ovh_m = [data[k]["mean_ms"] - base for k in LAT_KEYS]
    ovh_s = [data[k]["std_ms"]         for k in LAT_KEYS]

    _bar(ax, x, [base] * len(x), [0] * len(x), DET_COLOR, DET_HATCH)

    for i, (c, h, om, os) in enumerate(
            zip(LAT_OVH_COLOR, LAT_OVH_HATCH, ovh_m, ovh_s)):
        total = base + om
        if om > 0.05:
            _bar(ax, [x[i]], [om], [os], c, h, bottom=base)
            ax.text(x[i], base + om + os + 2.5, f"{total:.1f}",
                    ha="center", va="bottom",
                    fontsize=LABEL_FS - 1, fontweight="bold")
        else:
            ax.text(x[i], base + 2.5, f"{total:.1f}",
                    ha="center", va="bottom",
                    fontsize=LABEL_FS - 1, fontweight="bold")

    ax.set_ylabel("Per-Frame Latency (ms)")
    top_max = base + max(ovh_m) + max(ovh_s)
    ax.set_ylim(0, top_max * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(LAT_LABELS, fontsize=LABEL_FS)
    ax.set_xlim(-0.6, len(x) - 0.4)
    _style_l(ax, "black")
    _legend(ax, [
        _patch(DET_COLOR, DET_HATCH, "Detector base"),
        _patch(GPU_COLOR, GPU_HATCH,  "GPU est. overhead"),
        _patch(PTC_COLOR, PTC_HATCH,  "PTC overhead (CPU)"),
        _patch(XGB_COLOR, XGB_HATCH,  "XGBoost overhead (CPU)"),
    ], ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Throughput (FPS) — GPU=blue, CPU=green, + detector-only ref line
# ─────────────────────────────────────────────────────────────────────────────
def plot_task2(ax, data, det_fps=None):
    x     = np.arange(len(THR_KEYS), dtype=float)
    means = [data[k]["fps"] for k in THR_KEYS]

    for i, (is_cpu, m) in enumerate(zip(THR_IS_CPU, means)):
        c, h = (PTC_COLOR, PTC_HATCH) if is_cpu else (GPU_COLOR, GPU_HATCH)
        _bar(ax, [x[i]], [m], [0], c, h)
        ax.text(x[i], m + 0.07, f"{m:.2f}",
                ha="center", va="bottom",
                fontsize=LABEL_FS - 1, fontweight="bold")

    if det_fps is not None:
        _refline(ax, det_fps, f"Det-Only ≈{det_fps:.2f} FPS")

    ax.set_ylabel("Throughput (FPS)")
    ax.set_ylim(0, max(means) * 1.38)
    ax.set_xticks(x)
    ax.set_xticklabels(THR_LABELS, fontsize=LABEL_FS)
    ax.set_xlim(-0.6, len(x) - 0.4)
    _style_l(ax, "black")
    _legend(ax, [_patch(GPU_COLOR, GPU_HATCH, "GPU estimator"),
                 _patch(PTC_COLOR, PTC_HATCH,  "CPU estimator")])


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — two sub-panels
#
#  Left  (plot_task3_delta):  CPU% delta + RAM overhead vs detector-only
#                             Source: resources.json  (has detector_only baseline)
#
#  Right (plot_task3_gpu):    GPU% absolute
#                             Source: resources_throughput.json
#                             (detector-only not captured in this run)
# ─────────────────────────────────────────────────────────────────────────────

# 4-condition x-axis shared by left panel (latency-run conditions)
RES_LAT_LABELS = ["IoUNet\n(GPU)", "GFL\n(GPU)", "PTC\n(CPU)", "XGBoost\n(CPU)"]
RES_LAT_KEYS   = ["iounet_gpu",    "gfl_gpu",    "ptc_cpu",    "xgboost_cpu"]
RES_LAT_COLOR  = [GPU_COLOR, GPU_COLOR, PTC_COLOR, XGB_COLOR]
RES_LAT_HATCH  = [GPU_HATCH, GPU_HATCH, PTC_HATCH, XGB_HATCH]


def _delta_bar(ax, xi, v, c, h, width=BAR_W2, offset=0.0):
    """Single bar that can go negative; returns the Rectangle artist."""
    bar = ax.bar(xi + offset, v, width=width,
                 color=c, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH,
                 hatch=h, zorder=3)
    return bar[0]


def plot_task3_delta(ax, res_lat,
                     right_ylabel="RAM Overhead vs Det-Only (MB)"):
    axr = ax.twinx()
    x   = np.arange(len(RES_LAT_KEYS), dtype=float)

    base_cpu = res_lat["detector_only"]["cpu_pct_mean"]
    base_ram = res_lat["detector_only"]["ram_mb_mean"]

    cpu_d = [res_lat[k]["cpu_pct_mean"] - base_cpu for k in RES_LAT_KEYS]
    ram_d = [res_lat[k]["ram_mb_mean"]  - base_ram for k in RES_LAT_KEYS]
    ram_s = [res_lat[k]["ram_mb_std"]              for k in RES_LAT_KEYS]

    for i, (v, c, h) in enumerate(zip(cpu_d, RES_LAT_COLOR, RES_LAT_HATCH)):
        _delta_bar(ax, x[i], v, c, h, offset=-BAR_W2 / 2)
        va = "bottom" if v >= 0 else "top"
        dy = 0.08 if v >= 0 else -0.08
        ax.text(x[i] - BAR_W2 / 2, v + dy, f"{v:+.1f}%",
                ha="center", va=va, fontsize=LABEL_FS - 1, fontweight="bold")

    ax.axhline(0, color=CPU_COLOR, linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)

    bs_ram = _bar(axr, x, ram_d, ram_s, RAM_COLOR, RAM_HATCH,
                  width=BAR_W2, offset=+BAR_W2 / 2)
    for b, v in zip(bs_ram, ram_d):
        _lbl(axr, b, v, fmt="+{:.0f}", dy=3)

    axr.axhline(0, color=RAM_COLOR, linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)

    cpu_abs = max(abs(v) for v in cpu_d)
    ax.set_ylim(-cpu_abs * 3.5, cpu_abs * 2.5)
    _ylim(axr, ram_d, ram_s, pad=1.6)

    ax.set_ylabel("CPU Util Change vs Det-Only (%)")
    axr.set_ylabel(right_ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(RES_LAT_LABELS, fontsize=LABEL_FS)
    ax.set_xlim(-0.6, len(x) - 0.4)
    _style_l(ax,  CPU_COLOR)
    _style_r(axr, RAM_COLOR)
    _legend(ax, [_patch(GPU_COLOR, GPU_HATCH, "GPU estimator"),
                 _patch(PTC_COLOR, PTC_HATCH, "PTC (CPU)"),
                 _patch(XGB_COLOR, XGB_HATCH, "XGBoost (CPU)"),
                 _patch(RAM_COLOR, RAM_HATCH,  "RAM overhead (MB)")], ncol=2)


def plot_task3_gpu(ax, res_thr):
    x     = np.arange(len(THR_KEYS), dtype=float)
    gpu_m = [res_thr[k]["gpu_pct_mean"] for k in THR_KEYS]
    gpu_s = [res_thr[k]["gpu_pct_std"]  for k in THR_KEYS]

    for i, (m, s, is_cpu) in enumerate(zip(gpu_m, gpu_s, THR_IS_CPU)):
        c, h = (PTC_COLOR, PTC_HATCH) if is_cpu else (GPU_COLOR, GPU_HATCH)
        bs = _bar(ax, [x[i]], [m], [s], c, h, width=BAR_W2)
        _lbl(ax, bs[0], m, dy=2.0)

    ax.set_ylabel("GPU Utilization (%)")
    _ylim(ax, gpu_m, gpu_s, pad=1.2, hi=100)
    ax.set_xticks(x)
    ax.set_xticklabels(THR_LABELS, fontsize=LABEL_FS)
    ax.set_xlim(-0.6, len(x) - 0.4)
    _style_l(ax, GPU_COLOR)
    _legend(ax, [_patch(GPU_COLOR, GPU_HATCH, "GPU estimator"),
                 _patch(PTC_COLOR, PTC_HATCH, "CPU estimator")])


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Power draw (left) + energy overhead % vs detector-only (right)
# ─────────────────────────────────────────────────────────────────────────────
def plot_task4(ax, data, right_ylabel="Energy Overhead vs Det-Only (%)"):
    axr   = ax.twinx()
    x     = np.arange(len(LAT_KEYS), dtype=float)
    pwr_m = [data[k]["power_mean_mw"]       for k in LAT_KEYS]
    pwr_s = [data[k]["power_std_mw"]        for k in LAT_KEYS]
    ovh   = [data[k]["energy_overhead_pct"] for k in LAT_KEYS]

    bs_pwr = _bar(ax, x, pwr_m, pwr_s, PWR_COLOR, PWR_HATCH,
                  width=BAR_W2, offset=-BAR_W2 / 2)
    for b, m in zip(bs_pwr, pwr_m):
        _lbl(ax, b, m, fmt="{:.0f}")

    abs_max = max(abs(v) for v in ovh)
    axr.set_ylim(-abs_max * 2.6, abs_max * 2.6)
    axr.axhline(0, color=ENE_COLOR, linewidth=0.9, linestyle="--",
                alpha=0.7, zorder=2)

    for i, v in enumerate(ovh):
        axr.bar(x[i] + BAR_W2 / 2, v, width=BAR_W2,
                color=ENE_COLOR, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH,
                hatch=ENE_HATCH, zorder=3)
        txt_y = v + (0.35 if v >= 0 else -0.35)
        va    = "bottom" if v >= 0 else "top"
        axr.text(x[i] + BAR_W2 / 2, txt_y, f"{v:+.1f}%",
                 ha="center", va=va,
                 fontsize=LABEL_FS - 1, fontweight="bold", color="saddlebrown")

    ax.set_ylabel("Avg Power (mW)")
    axr.set_ylabel(right_ylabel)
    _ylim(ax, pwr_m, pwr_s, pad=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(LAT_LABELS, fontsize=LABEL_FS)
    ax.set_xlim(-0.6, len(x) - 0.4)
    _style_l(ax,  PWR_COLOR)
    _style_r(axr, ENE_COLOR)
    _legend(ax, [_patch(PWR_COLOR, PWR_HATCH, "Power (mW)"),
                 _patch(ENE_COLOR, ENE_HATCH,  "Energy overhead (%)")])


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def save_task1(output=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.76, bottom=0.22)
    plot_task1(ax, load("latency.json"))
    _save(fig, output or RESULTS_DIR / "task1_latency.pdf")


def save_task2(output=None):
    lat     = load("latency.json")
    det_fps = 1000.0 / lat["detector_only"]["mean_ms"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.22)
    plot_task2(ax, load("throughput.json"), det_fps)
    _save(fig, output or RESULTS_DIR / "task2_throughput.pdf")


def save_task3(output=None):
    res_lat = load("resources.json")
    res_thr = load("resources_throughput.json")
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 4.2))
    fig.subplots_adjust(left=0.09, right=0.93, top=0.80, bottom=0.22, wspace=0.60)
    plot_task3_delta(ax_l, res_lat)
    plot_task3_gpu(ax_r, res_thr)
    _save(fig, output or RESULTS_DIR / "task3_resources.pdf")


def save_task4(output=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.subplots_adjust(left=0.13, right=0.86, top=0.82, bottom=0.22)
    plot_task4(ax, load("energy.json"))
    _save(fig, output or RESULTS_DIR / "task4_energy.pdf")


def save_combined(output=None):
    """2x2 combined figure; Task 3 cell split into two sub-panels via GridSpec."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    lat_data = load("latency.json")
    thr_data = load("throughput.json")
    res_lat  = load("resources.json")
    res_thr  = load("resources_throughput.json")
    ene_data = load("energy.json")
    det_fps  = 1000.0 / lat_data["detector_only"]["mean_ms"]

    fig = plt.figure(figsize=(18, 8))
    outer = GridSpec(2, 2, figure=fig,
                     wspace=0.52, hspace=0.72,
                     left=0.06, right=0.96, top=0.88, bottom=0.13)

    ax1  = fig.add_subplot(outer[0, 0])
    ax2  = fig.add_subplot(outer[0, 1])
    ax4  = fig.add_subplot(outer[1, 1])

    # Task 3 cell → split into left (delta) + right (GPU%)
    inner3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.70)
    ax3l   = fig.add_subplot(inner3[0])
    ax3r   = fig.add_subplot(inner3[1])

    plot_task1(ax1, lat_data)
    plot_task2(ax2, thr_data, det_fps)
    plot_task3_delta(ax3l, res_lat)
    plot_task3_gpu(ax3r, res_thr)
    plot_task4(ax4, ene_data)

    _save(fig, output or RESULTS_DIR / "tasks_all.pdf")


def save_task3_task4(output=None):
    """1×3 combined figure: CPU+RAM delta | GPU% | Power+Energy overhead."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    res_lat  = load("resources.json")
    res_thr  = load("resources_throughput.json")
    ene_data = load("energy.json")

    fig = plt.figure(figsize=(18.5, 4.5))
    outer = GridSpec(1, 2, figure=fig,
                     width_ratios=[1.0, 2.6],
                     wspace=0.2,
                     left=0.05, right=0.94, top=0.80, bottom=0.30)
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                    width_ratios=[1.5, 1.1],
                                    wspace=0.15)

    ax1 = fig.add_subplot(outer[0])
    ax2 = fig.add_subplot(inner[0])
    ax3 = fig.add_subplot(inner[1])

    plot_task3_delta(ax1, res_lat, right_ylabel="RAM (MB)")
    plot_task3_gpu(ax2, res_thr)
    plot_task4(ax3, ene_data, right_ylabel="Energy Overhead (%)")

    for ax, title in [
        (ax1, "(a) CPU Util. & RAM Overhead"),
        (ax2, "(b) GPU Utilization"),
        (ax3, "(c) Power & Energy Overhead"),
    ]:
        ax.text(0.5, -0.32, title,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=FONTSIZE, fontweight="bold")

    _save(fig, output or RESULTS_DIR / "task3_task4_combined.pdf")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot Tasks 1–4 results")
    parser.add_argument("--task",     type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--output",   type=str)
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()

    savers = {1: save_task1, 2: save_task2, 3: save_task3, 4: save_task4}
    if args.task:
        savers[args.task](args.output)
    elif args.combined:
        save_combined(args.output)
    else:
        for fn in savers.values():
            fn()
        save_task3_task4()
        save_combined(args.output)


if __name__ == "__main__":
    main()
