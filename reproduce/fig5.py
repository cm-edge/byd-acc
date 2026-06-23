#!/usr/bin/env python3
"""
Reproduce Appendix C Figure 5 — Achieved mAP vs Offloading Ratio.

Three-panel figure (UA-DETRAC / UAVDT / MOT17), each showing mAP curves for:
  PTC-IoU, XGBoost, IoU-Net, GFL
  plus Random, Oracle, Weak-only, Strong-only reference lines.

The percentage beside each curve is Oracle Gap @ 20% budget:
  OG@20% = (achieved - weak) / (oracle - weak)  at ρ=0.2

Requirements:
  - scores/ populated by scoring/compute_ptc.py, scoring/compute_iounet_gfl.py,
    scoring/compute_xgboost.py
  - scores/derived/{dataset}/{video}.csv with columns frame_id, weak_map, strong_map

Output:
  reproduce/output/fig5.pdf
  reproduce/output/fig5.png

Usage:
    python reproduce/fig5.py
    python reproduce/fig5.py --output-dir /tmp/figs
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dataset"))

SCORES_DIR  = ROOT / "scores"
SPLITS_FILE = ROOT / "dataset" / "video_splits.json"
OUT_DIR     = Path(__file__).parent / "output"

# Per-dataset method configurations (matches plot_final_comparison.py):
#   UA-DETRAC: mean scores, invert=True (high score = good quality = don't offload)
#   UAVDT: top3 scores, invert=False (model anti-correlated due to domain shift)
#   MOT17: mean scores, invert=True
STYLE = {
    "PTC-IoU": {"color": "#d62728", "marker": "D", "ls": "-"},
    "XGBoost": {"color": "#ff7f0e", "marker": "^", "ls": "-"},
    "IoU-Net": {"color": "#2ca02c", "marker": "s", "ls": "-"},
    "GFL":     {"color": "#9467bd", "marker": "o", "ls": "-"},
}

DS_METHODS = {
    "ua-detrac": {
        "IoU-Net": {"dir": "iounet",  "col": "iou_net_mean",    "invert": True},
        "GFL":     {"dir": "gfl",     "col": "gfl_mean",        "invert": True},
        "PTC-IoU": {"dir": "ptc",     "col": "ptc_h",           "invert": True},
        "XGBoost": {"dir": "xgboost", "col": "predicted_gain",  "invert": False},
    },
    "uavdt": {
        "IoU-Net": {"dir": "iounet",  "col": "iou_net_top3",    "invert": False},
        "GFL":     {"dir": "gfl",     "col": "gfl_top3",        "invert": False},
        "PTC-IoU": {"dir": "ptc",     "col": "ptc_h",           "invert": False},
        "XGBoost": {"dir": "xgboost", "col": "predicted_gain",  "invert": False},
    },
    "mot17": {
        "IoU-Net": {"dir": "iounet",  "col": "iou_net_mean",    "invert": True},
        "GFL":     {"dir": "gfl",     "col": "gfl_mean",        "invert": True},
        "PTC-IoU": {"dir": "ptc",     "col": "ptc_h",           "invert": True},
        "XGBoost": {"dir": "xgboost", "col": "predicted_gain",  "invert": False},
    },
}

DATASETS = [
    ("ua-detrac", "UA-DETRAC"),
    ("uavdt",     "UAVDT"),
    ("mot17",     "MOT17"),
]

OFFLOAD_STEPS = np.linspace(0, 1, 21)

# MOT17 global-AP routing results from scoring/compute_mot17_routing.py
MOT17_ROUTING_CSV  = SCORES_DIR / "mot17_routing_curves.csv"
MOT17_PAIRWISE_CSV = SCORES_DIR / "mot17_routing_pairwise.csv"


def _load_video_data(dataset: str, videos: list):
    """Return list of (weak_map_arr, strong_map_arr) per video."""
    derived = SCORES_DIR / "derived" / dataset
    out = []
    for v in videos:
        p = derived / f"{v}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).sort_values("frame_id")
        if "weak_map" not in df.columns or "strong_map" not in df.columns:
            continue
        out.append((df["weak_map"].values, df["strong_map"].values,
                    df["frame_id"].values))
    return out


def _load_scores(dataset: str, method_cfg: dict, videos: list):
    score_dir = SCORES_DIR / method_cfg["dir"] / dataset
    col = method_cfg["col"]
    out = {}
    for v in videos:
        p = score_dir / f"{v}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if col in df.columns:
            out[v] = df.set_index("frame_id")[col]
    return out


def simulate_offloading(video_data, score_map: dict, videos: list,
                         invert: bool, rho: float) -> float:
    all_weak, all_strong, all_scores = [], [], []
    for (v, (weak, strong, fids)) in zip(videos, video_data):
        if v not in score_map:
            continue
        s = score_map[v]
        idx = np.array([i for i, f in enumerate(fids) if f in s.index])
        if len(idx) == 0:
            continue
        all_weak.append(weak[idx])
        all_strong.append(strong[idx])
        sc = np.array([s.loc[f] for f in fids[idx]])
        all_scores.append(sc)

    if not all_weak:
        return float("nan")

    w = np.concatenate(all_weak)
    s = np.concatenate(all_strong)
    sc = np.concatenate(all_scores)
    n = len(w)
    k = max(1, int(round(rho * n)))

    rank = np.argsort(sc if invert else -sc)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[rank] = True

    achieved = float(np.mean(np.where(mask, s, w)))
    return achieved


def oracle_line(video_data, rho: float) -> float:
    all_weak, all_strong = [], []
    for (w, s, _) in video_data:
        all_weak.append(w)
        all_strong.append(s)
    w = np.concatenate(all_weak)
    s = np.concatenate(all_strong)
    n = len(w)
    k = max(1, int(round(rho * n)))
    delta = s - w
    top_k = np.argsort(delta)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[top_k] = True
    return float(np.mean(np.where(mask, s, w)))


def og20(achieved_curve, weak_ap, oracle_curve):
    idx20 = int(round(0.2 * (len(OFFLOAD_STEPS) - 1)))
    a = achieved_curve[idx20]
    o = oracle_curve[idx20]
    if o - weak_ap < 1e-9:
        return 0.0
    return (a - weak_ap) / (o - weak_ap)


def plot_mot17_from_csv(ax, label: str):
    """Plot MOT17 panel using global-AP routing curves from precomputed CSV."""
    if not MOT17_ROUTING_CSV.exists():
        ax.text(0.5, 0.5, f"Run\nscoring/compute_mot17_routing.py",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(label)
        return

    curves   = pd.read_csv(MOT17_ROUTING_CSV)
    pairwise = pd.read_csv(MOT17_PAIRWISE_CSV) if MOT17_PAIRWISE_CSV.exists() else None

    ratios = curves["offload_ratio"].values / 100.0
    pct    = ratios * 100

    weak_ap   = float(curves.loc[curves["offload_ratio"] == 0, "weak"].iloc[0])
    strong_ap = float(curves.loc[curves["offload_ratio"] == 100, "strong"].iloc[0])

    ax.plot(pct, [weak_ap]   * len(pct), "k:", lw=1, label=f"Weak only ({weak_ap:.4f})")
    ax.plot(pct, [strong_ap] * len(pct), "k--", lw=1, label=f"Strong only ({strong_ap:.4f})")
    ax.plot(pct, curves["oracle"].values, "k-", lw=1.5, alpha=0.4, label="Oracle")
    ax.plot(pct, curves["random"].values, color="gray", lw=1, ls="--", label="Random")

    idx20 = np.argmin(np.abs(ratios - 0.2))
    oracle20 = float(curves["oracle"].values[idx20])

    for name, cfg in DS_METHODS["mot17"].items():
        if name not in curves.columns:
            continue
        achieved = curves[name].values
        a20 = float(achieved[idx20])
        og  = (a20 - weak_ap) / (oracle20 - weak_ap) if oracle20 - weak_ap > 1e-9 else 0.0
        st  = STYLE[name]
        ax.plot(pct, achieved, color=st["color"], marker=st["marker"],
                markersize=3, ls=st["ls"], lw=1.5, label=f"{name} ({og*100:.1f}%)")

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Offloading Ratio (%)")
    ax.set_ylabel("Achieved mAP")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_dataset(ax, dataset: str, videos: list, label: str):
    video_data = _load_video_data(dataset, videos)
    if not video_data:
        ax.text(0.5, 0.5, f"No derived data\nfor {dataset}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        return

    w_all = np.concatenate([vd[0] for vd in video_data])
    s_all = np.concatenate([vd[1] for vd in video_data])
    weak_ap   = float(np.mean(w_all))
    strong_ap = float(np.mean(s_all))

    oracle_curve = [oracle_line(video_data, r) for r in OFFLOAD_STEPS]

    pct = OFFLOAD_STEPS * 100
    ax.plot(pct, [weak_ap] * len(pct),
            "k:", lw=1, label=f"Weak only ({weak_ap:.4f})")
    ax.plot(pct, [strong_ap] * len(pct),
            "k--", lw=1, label=f"Strong only ({strong_ap:.4f})")
    ax.plot(pct, oracle_curve,
            "k-", lw=1.5, alpha=0.4, label="Oracle")

    random_curve = [weak_ap + r * (strong_ap - weak_ap) for r in OFFLOAD_STEPS]
    ax.plot(pct, random_curve,
            color="gray", lw=1, ls="--", label="Random")

    ds_methods = DS_METHODS.get(dataset, DS_METHODS["ua-detrac"])
    for name, cfg in ds_methods.items():
        score_map = _load_scores(dataset, cfg, videos)
        if not score_map:
            continue
        curve = [simulate_offloading(video_data, score_map, videos,
                                      cfg["invert"], r)
                 for r in OFFLOAD_STEPS]
        og = og20(curve, weak_ap, oracle_curve)
        st = STYLE[name]
        ax.plot(pct, curve,
                color=st["color"], marker=st["marker"], markersize=3,
                ls=st["ls"], lw=1.5,
                label=f"{name} ({og*100:.1f}%)")

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Offloading Ratio (%)")
    ax.set_ylabel("Achieved mAP")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Fig 5: Achieved mAP vs Offloading Ratio\n"
        "FASTERRCNN-MNv3L-320 → FASTERRCNN-R50-FPNv2",
        fontsize=10,
    )

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        if ds_key == "mot17":
            plot_mot17_from_csv(ax, ds_label)
        else:
            videos = splits.get(ds_key, {}).get("test", [])
            plot_dataset(ax, ds_key, videos, ds_label)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig5.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
