#!/usr/bin/env python3
"""
Reproduce Table 2 — Estimation quality across three datasets at ρ=0.3.

Pipeline (matches paper's plot_paper_final.py + plot_final_comparison.py):

  UA-DETRAC / UAVDT / MOT17:
    - Estimator scores: iounet/gfl/ptc scores from scores/ directory
    - Columns used: iou_net_mean (UA-DETRAC/MOT17, invert=True),
                    iou_net_top3 (UAVDT, invert=False)
                    gfl_mean (UA-DETRAC/MOT17, True), gfl_top3 (UAVDT, False)
                    ptc_h (all datasets, True for UA-DETRAC/MOT17, False for UAVDT)
                    predicted_gain (XGBoost, all, invert=False)
    - AP metric: mean per-frame AP (weak_map/strong_map from scores/derived/)
    - Test videos: from dataset/video_splits.json

  Note: MOT17 paper numbers use global detection AP, while this script uses mean
  per-frame AP (consistent with UA-DETRAC/UAVDT). Run scoring scripts first:
    python scoring/compute_iounet_gfl.py --datasets mot17
    python scoring/compute_ptc.py --datasets mot17

Usage:
    python reproduce/tab2.py
    python reproduce/tab2.py --offload-ratio 0.3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dataset"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORES_DIR   = ROOT / "scores"
SPLITS_FILE  = ROOT / "dataset" / "video_splits.json"

# Score column + invert convention (from plot_final_comparison.py)
# invert=True  → high score means good quality (don't offload) → negate to sort worst-first
# invert=False → high score means offload this frame
SCORE_CFG = {
    "ua-detrac": {
        "IoU-Net": (SCORES_DIR/"iounet", "iou_net_mean",    True),
        "GFL":     (SCORES_DIR/"gfl",    "gfl_mean",        True),
        "PTC-IoU": (SCORES_DIR/"ptc",    "ptc_h",           True),
        "XGBoost": (SCORES_DIR/"xgboost","predicted_gain",  False),
    },
    "uavdt": {
        "IoU-Net": (SCORES_DIR/"iounet", "iou_net_top3",    False),
        "GFL":     (SCORES_DIR/"gfl",    "gfl_top3",        False),
        "PTC-IoU": (SCORES_DIR/"ptc",    "ptc_h",           False),
        "XGBoost": (SCORES_DIR/"xgboost","predicted_gain",  False),
    },
    "mot17": {
        # Matches original plot_final_comparison.py: iou_net_mean, gfl_mean
        # ptc_h: ptc_ass * ptc_loc^0.15  (grid-searched optimal, cfg_id=698)
        "IoU-Net": (SCORES_DIR/"iounet", "iou_net_mean",    True),
        "GFL":     (SCORES_DIR/"gfl",    "gfl_mean",        True),
        "PTC-IoU": (SCORES_DIR/"ptc",    "ptc_h",           True),
        "XGBoost": (SCORES_DIR/"xgboost","predicted_gain",  False),
    },
}

OFFLOAD_RATIOS = list(range(0, 101, 10))   # 0..100 step 10

# MOT17 global-AP routing results (produced by scoring/compute_mot17_routing.py)
MOT17_ROUTING_CSV  = SCORES_DIR / "mot17_routing_curves.csv"
MOT17_PAIRWISE_CSV = SCORES_DIR / "mot17_routing_pairwise.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pairwise_acc(scores: np.ndarray, gains: np.ndarray,
                 max_samples: int = 6000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        scores, gains = scores[idx], gains[idx]
        n = max_samples
    ii, jj = np.triu_indices(n, k=1)
    mask = gains[ii] != gains[jj]
    if not mask.sum():
        return float("nan")
    si, sj = scores[ii[mask]], scores[jj[mask]]
    gi, gj = gains[ii[mask]], gains[jj[mask]]
    concordant = np.sum((si > sj) == (gi > gj)) + 0.5 * np.sum(si == sj)
    return float(concordant) / int(mask.sum())


def og20(achieved_curve: list, oracle_curve: list, weak_ap: float) -> float:
    if len(achieved_curve) < 3:
        return float("nan")
    idx = 2  # ρ=20% is index 2 in OFFLOAD_RATIOS (0,10,20,...)
    og = oracle_curve[idx] - weak_ap
    if og < 1e-9:
        return 0.0
    return (achieved_curve[idx] - weak_ap) / og * 100.0


# ---------------------------------------------------------------------------
# MOT17 global-AP routing: read from precomputed CSVs
# ---------------------------------------------------------------------------
def _load_mot17_routing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (curves_df, pairwise_df) or (None, None) if files missing."""
    if not MOT17_ROUTING_CSV.exists() or not MOT17_PAIRWISE_CSV.exists():
        return None, None
    curves = pd.read_csv(MOT17_ROUTING_CSV)    # offload_ratio, oracle, random, weak, strong, IoU-Net, GFL, PTC-IoU, XGBoost
    pairwise = pd.read_csv(MOT17_PAIRWISE_CSV)  # method, pairwise_acc, og20
    return curves, pairwise


def mot17_baselines_from_csv(rho: float = 0.3) -> Tuple[float, float, float, float]:
    curves, _ = _load_mot17_routing()
    if curves is None:
        return tuple([float("nan")] * 4)
    row = curves[curves["offload_ratio"] == int(round(rho * 100))].iloc[0]
    return float(row["weak"]), float(row["strong"]), float(row["random"]), float(row["oracle"])


def mot17_eval_from_csv(method: str, rho: float = 0.3) -> Tuple[float, float]:
    curves, pairwise = _load_mot17_routing()
    if curves is None:
        return float("nan"), float("nan")
    pct = int(round(rho * 100))
    row = curves[curves["offload_ratio"] == pct]
    if row.empty or method not in curves.columns:
        return float("nan"), float("nan")
    ap = float(row.iloc[0][method])
    pw_row = pairwise[pairwise["method"] == method]
    pw = float(pw_row["pairwise_acc"].values[0]) if not pw_row.empty else float("nan")
    return ap, pw


# ---------------------------------------------------------------------------
# UA-DETRAC / UAVDT evaluation (mean per-frame AP routing)
# ---------------------------------------------------------------------------
def load_derived(dataset: str, videos: List[str]) -> Dict[str, Tuple]:
    """Return {video: (weak_arr, strong_arr, frame_ids)}."""
    derived = SCORES_DIR / "derived" / dataset
    out = {}
    for v in videos:
        p = derived / f"{v}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "weak_map" not in df.columns:
            continue
        out[v] = (df["weak_map"].values, df["strong_map"].values,
                  df["frame_id"].values)
    return out


def load_scores_col(sdir: Path, col: str, videos: List[str],
                    dataset: str) -> Dict[str, pd.Series]:
    out = {}
    for v in videos:
        p = sdir / dataset / f"{v}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if col not in df.columns:
            continue
        out[v] = df.set_index("frame_id")[col]
    return out


def simulate_offloading(score_map: dict, derived: dict, videos: list,
                         invert: bool, rho: float) -> float:
    """Matches plot_final_comparison.py _curve() logic."""
    all_w, all_s, all_sc = [], [], []
    for v in videos:
        if v not in score_map or v not in derived:
            continue
        s = score_map[v]
        w_arr, s_arr, fids = derived[v]
        idx = np.array([i for i, f in enumerate(fids) if f in s.index])
        if not len(idx):
            continue
        raw_sc = np.array([s.loc[f] for f in fids[idx]])
        if invert:
            raw_sc = -raw_sc   # high=offload-first after negation
        all_sc.append(raw_sc)
        all_w.append(w_arr[idx])
        all_s.append(s_arr[idx])

    if not all_sc:
        return float("nan")
    sc = np.concatenate(all_sc)
    w  = np.concatenate(all_w)
    st = np.concatenate(all_s)
    n  = len(sc)
    k  = max(1, int(round(rho * n)))
    rank = np.argsort(sc)[::-1][:k]
    mask = np.zeros(n, dtype=bool)
    mask[rank] = True
    return float(np.mean(np.where(mask, st, w)))


def eval_dataset(method: str, dataset: str, videos: List[str],
                  rho: float = 0.3) -> Tuple[float, float]:
    """Unified routing evaluation for all datasets using actual score files."""
    if dataset not in SCORE_CFG or method not in SCORE_CFG[dataset]:
        return float("nan"), float("nan")
    sdir, col, invert = SCORE_CFG[dataset][method]
    scores  = load_scores_col(sdir, col, videos, dataset)
    derived = load_derived(dataset, videos)

    if not scores or not derived:
        return float("nan"), float("nan")

    # Routing AP
    ap = simulate_offloading(scores, derived, videos, invert, rho)

    # Pairwise accuracy
    all_sc, all_delta = [], []
    for v in scores:
        if v not in derived:
            continue
        s = scores[v]
        w_arr, s_arr, fids = derived[v]
        idx = np.array([i for i, f in enumerate(fids) if f in s.index])
        if not len(idx):
            continue
        raw_sc = np.array([s.loc[f] for f in fids[idx]])
        if invert:
            raw_sc = -raw_sc
        all_sc.append(raw_sc)
        all_delta.append(s_arr[idx] - w_arr[idx])

    if not all_sc:
        return ap, float("nan")
    pw = pairwise_acc(np.concatenate(all_sc), np.concatenate(all_delta))
    return ap, pw


def get_baselines(dataset: str, videos: List[str],
                   rho: float = 0.3) -> Tuple[float, float, float, float]:
    """Weak/Strong/Random/Oracle baselines from per-frame AP for all datasets."""
    derived = load_derived(dataset, videos)
    if not derived:
        return tuple([float("nan")] * 4)
    w  = np.concatenate([derived[v][0] for v in derived])
    st = np.concatenate([derived[v][1] for v in derived])
    weak_ap   = float(np.mean(w))
    strong_ap = float(np.mean(st))
    n = len(w); k = max(1, int(round(rho * n)))
    rng = np.random.default_rng(42)
    m = np.zeros(n, bool); m[rng.choice(n, k, replace=False)] = True
    rand_ap = float(np.mean(np.where(m, st, w)))
    delta = st - w
    om = np.zeros(n, bool); om[np.argsort(delta)[::-1][:k]] = True
    oracle_ap = float(np.mean(np.where(om, st, w)))
    return weak_ap, strong_ap, rand_ap, oracle_ap


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------
def get_test_videos(dataset: str, splits: dict) -> List[str]:
    """For UA-DETRAC: all official test-split videos that have derived data."""
    if dataset == "ua-detrac":
        from gt_parser import get_video_split
        derived = SCORES_DIR / "derived" / dataset
        if derived.exists():
            return sorted(
                p.stem for p in derived.glob("MVI_*.csv")
                if get_video_split(p.stem) == "test"
            )
        return splits[dataset]["test"]
    return splits[dataset]["test"]


def evaluate(method: str, dataset: str, videos: List[str],
             rho: float = 0.3) -> Tuple[float, float]:
    if dataset == "mot17":
        return mot17_eval_from_csv(method, rho)
    return eval_dataset(method, dataset, videos, rho)


def baselines(dataset: str, videos: List[str],
              rho: float = 0.3) -> Tuple[float, float, float, float]:
    if dataset == "mot17":
        return mot17_baselines_from_csv(rho)
    return get_baselines(dataset, videos, rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-ratio", type=float, default=0.3)
    parser.add_argument("--datasets", nargs="+",
                        default=["ua-detrac", "uavdt", "mot17"])
    args = parser.parse_args()
    rho = args.offload_ratio

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    PAPER = {
        "ua-detrac": {"IoU-Net":(0.682,58.2),"GFL":(0.679,56.7),
                      "PTC-IoU":(0.680,55.3),"XGBoost":(0.682,57.2)},
        "uavdt":     {"IoU-Net":(0.194,58.5),"GFL":(0.227,64.4),
                      "PTC-IoU":(0.214,57.5),"XGBoost":(0.231,69.0)},
        "mot17":     {"IoU-Net":(0.318,75.5),"GFL":(0.318,74.4),
                      "PTC-IoU":(0.313,70.3),"XGBoost":(0.299,53.1)},
    }
    DS_LABEL = {"ua-detrac":"UA-DETRAC","uavdt":"UAVDT","mot17":"MOT17"}
    # Paper baselines: MOT17 uses global AP in paper (0.253 weak), but our pipeline
    # computes mean per-frame AP (~0.336). The per-frame AP baseline is shown instead.
    PAPER_BL = {
        "ua-detrac":{"weak":0.636,"strong":0.748},
        "uavdt":    {"weak":0.119,"strong":0.336},
        "mot17":    {"weak":0.253,"strong":0.379},
    }

    print(f"\nTable 2 — Estimation quality at ρ={rho}")
    W = 24
    hdr = f"{'Method':<10}"
    for ds in args.datasets:
        hdr += f"  {DS_LABEL[ds]:>{W}}"
    print(hdr)
    print("-" * (10 + len(args.datasets) * (W + 2)))
    sub = f"{'':10}"
    for _ in args.datasets:
        sub += f"  {'AP':>6}/paper  {'Pair':>6}/paper"
    print(sub)
    print("-" * (10 + len(args.datasets) * (W + 2)))

    for label, is_baseline, b_key in [
        ("Weak only",   True, "weak"),
        ("Strong only", True, "strong"),
    ]:
        row = f"{label:<10}"
        for ds in args.datasets:
            vids = get_test_videos(ds, splits)
            w, s, r, o = baselines(ds, vids, rho)
            val = w if b_key == "weak" else s
            pv  = PAPER_BL[ds][b_key]
            row += f"  {val:>6.3f}({pv:.3f})  {'n/a':>10}"
        print(row)
    print("-" * (10 + len(args.datasets) * (W + 2)))

    for method in ["IoU-Net", "GFL", "PTC-IoU", "XGBoost"]:
        row = f"{method:<10}"
        for ds in args.datasets:
            vids = get_test_videos(ds, splits)
            ap, pw = evaluate(method, ds, vids, rho)
            pap, ppw = PAPER[ds][method]
            ap_s = f"{ap:.3f}" if not np.isnan(ap) else " n/a"
            pw_s = f"{pw*100:.1f}%" if not np.isnan(pw) else "  n/a"
            row += f"  {ap_s:>6}({pap:.3f})  {pw_s:>6}({ppw:.1f}%)"
        print(row)

    print("-" * (10 + len(args.datasets) * (W + 2)))
    print("\nNote: Values shown as computed(paper).")
    if "mot17" in args.datasets:
        if not MOT17_ROUTING_CSV.exists():
            print("\n[WARN] MOT17 routing CSV not found. Run:")
            print("       python scoring/compute_mot17_routing.py")
        else:
            print(f"\n[INFO] MOT17 uses global AP routing from {MOT17_ROUTING_CSV.name}")


if __name__ == "__main__":
    main()
