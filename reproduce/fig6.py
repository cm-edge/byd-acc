#!/usr/bin/env python3
"""
Reproduce Appendix C Figure 6 — Frame-level quality vs routing gain on MOT17.

Scatter plot of weak-detector frame AP against normalised strong-model gain,
plus a binned-mean curve showing the moderate negative correlation.

Reproduces the two-panel figure (scatter top, binned mean bottom) with
Pearson and Spearman correlations reported.

Requirements:
  - scores/derived/mot17/{video}.csv with columns frame_id, weak_map, strong_map

Output:
  reproduce/output/fig6.pdf
  reproduce/output/fig6.png

Usage:
    python reproduce/fig6.py
    python reproduce/fig6.py --output-dir /tmp/figs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
SCORES_DIR  = ROOT / "scores"
SPLITS_FILE = ROOT / "dataset" / "video_splits.json"
OUT_DIR     = Path(__file__).parent / "output"


def load_mot17_frames(splits: dict) -> tuple:
    # scores/mot17_oracle_frame_scores.csv is produced by scoring/compute_mot17_routing.py.
    oracle_csv = SCORES_DIR / "mot17_oracle_frame_scores.csv"
    if oracle_csv.exists():
        df = pd.read_csv(oracle_csv)
        weak  = df["weak_ap_frame"].values
        delta = df["oracle_ap_frame_gain"].values
        max_abs_gain = np.max(np.abs(delta))
        gain_norm = delta / (max_abs_gain + 1e-12)
        return weak, gain_norm

    # Fallback: per-frame mean AP from derived/ (produces different correlation)
    videos  = splits.get("mot17", {}).get("test", [])
    derived = SCORES_DIR / "derived" / "mot17"
    all_weak, all_delta = [], []
    for v in videos:
        p = derived / f"{v}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "weak_map" not in df.columns or "strong_map" not in df.columns:
            continue
        all_weak.append(df["weak_map"].values)
        all_delta.append(df["strong_map"].values - df["weak_map"].values)

    if not all_weak:
        return None, None

    weak  = np.concatenate(all_weak)
    delta = np.concatenate(all_delta)
    max_abs_gain = np.max(np.abs(delta))
    gain_norm = delta / (max_abs_gain + 1e-12)
    return weak, gain_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    quality, gain = load_mot17_frames(splits)
    if quality is None:
        sys.exit(
            "No MOT17 derived data found in scores/derived/mot17/.\n"
            "Ensure weak and strong detection results are available."
        )

    pearson_r, _  = pearsonr(quality, gain)
    spearman_r, _ = spearmanr(quality, gain)
    n = len(quality)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 7),
                                          gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Fig 6: Frame-level quality vs routing gain (MOT17)", fontsize=10)

    # ── Scatter ───────────────────────────────────────────────────────────────
    density = np.ones(n)
    ax_top.scatter(quality, gain, c=density, cmap="viridis",
                   alpha=0.15, s=4, rasterized=True)

    # trend line
    coef = np.polyfit(quality, gain, 1)
    x_line = np.linspace(quality.min(), quality.max(), 200)
    ax_top.plot(x_line, np.polyval(coef, x_line), "b-", lw=1.5,
                label="Observed trend")
    ax_top.axhline(0, color="gray", lw=0.8, ls="--")

    ax_top.set_xlabel("Weak frame AP (quality)")
    ax_top.set_ylabel("Normalised strong-model gain")
    info = (f"Pearson = {pearson_r:.3f}\n"
            f"Spearman = {spearman_r:.3f}\n"
            f"# Frames = {n:,}")
    ax_top.text(0.97, 0.97, info, transform=ax_top.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax_top.legend(fontsize=8)

    # ── Binned mean ───────────────────────────────────────────────────────────
    bins = np.linspace(0, 1, args.n_bins + 1)
    bin_centers, bin_means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (quality >= lo) & (quality < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            bin_means.append(float(gain[mask].mean()))

    ax_bot.plot(bin_centers, bin_means, "k-o", ms=5, lw=1.5, label="Binned mean")
    ax_bot.axhline(0, color="gray", lw=0.8, ls="--")
    ax_bot.set_xlabel("Weak frame AP (quality)")
    ax_bot.set_ylabel("Binned mean\nnormalised gain")
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig6.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")

    print(f"\nPearson = {pearson_r:.3f}  Spearman = {spearman_r:.3f}  N={n:,}")


if __name__ == "__main__":
    main()
