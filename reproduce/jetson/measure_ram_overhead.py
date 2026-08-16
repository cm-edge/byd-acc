#!/usr/bin/env python3
"""
Clean per-estimator RAM-overhead measurement for Fig 3(a).

WHY THIS EXISTS
---------------
run_latency.py measures RAM with tegrastats (whole-system used RAM) sliced by
per-condition timing markers.  Two effects make its per-estimator RAM deltas
unreliable on a busy host:

  1. Baseline drift — detector_only and xgboost_cpu windows are seconds apart;
     background processes (vscode-server, browser, jtop) move system RAM by
     ±100-180 MB, which swamps and can flip the sign of a +60..140 MB estimator
     footprint (observed: XGBoost measured at +2 MB vs paper +141 MB, and whole
     rounds coming out negative).
  2. Shared imports — all 5 conditions run in ONE process, so the first-time
     import cost of sklearn/xgboost/scipy/pandas (~100 MB RSS) that *belongs* to
     XGBoost lands in the shared baseline instead of the XGBoost delta.

This script fixes both: it measures EACH condition in its own FRESH process and
takes the RAM delta over a SHORT interval (baseline right before loading the
estimator → right after), so background drift cannot accumulate and each
estimator is charged its full import + weight footprint.

On Jetson the CPU and GPU share one LPDDR pool (unified memory), so system
used-RAM (MemTotal - MemAvailable) is the correct figure and captures both the
GPU-resident neural-head weights and the CPU-side XGBoost libraries — exactly
what the paper's tegrastats RAM column measured, just without the drift.

We report three numbers per estimator so the source of each is auditable:
  used_ram_mb  — system used RAM delta  (headline; matches paper methodology)
  rss_mb       — this process RSS delta (CPU-side: xgboost libs + model)
  cuda_mb      — torch CUDA allocator delta (GPU-side: neural-head weights)

USAGE (inside the jetson-inference container, MAXN):
    python3 reproduce/jetson/measure_ram_overhead.py            # all conditions
    python3 reproduce/jetson/measure_ram_overhead.py --cond xgboost_cpu   # one (child)
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "jetson" / "benchmark" / "results"
CONDS = ["detector_only", "iounet_gpu", "gfl_gpu", "ptc_cpu", "xgboost_cpu"]


# ── system / process memory probes ────────────────────────────────────────────
def used_ram_mb() -> float:
    """System used RAM = MemTotal - MemAvailable (MB). Matches tegrastats 'used'."""
    total = avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) / 1024.0
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) / 1024.0
            if total is not None and avail is not None:
                break
    return total - avail


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def sample(n=12, dt=0.12):
    """Median system-used-RAM over a short window (robust to transient blips)."""
    vals = []
    for _ in range(n):
        vals.append(used_ram_mb())
        time.sleep(dt)
    vals.sort()
    return vals[len(vals) // 2]


def cuda_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def settle():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── child: measure ONE condition in a fresh process ───────────────────────────
def measure_one(cond: str) -> dict:
    # Reuse the exact loaders/constants the real benchmark uses.
    bench = ROOT / "jetson" / "benchmark"
    sys.path.insert(0, str(bench))
    import run_latency as RL  # noqa: E402  (module-level: imports torch/torchvision/estimators-min)

    # 1) Load the detector and warm CUDA — this is the shared baseline every
    #    condition starts from.
    detector, _ = RL.load_detector()
    # a couple of tiny dummy forwards to force CUDA context / cudnn workspaces
    import torch
    with torch.no_grad():
        for _ in range(3):
            detector([torch.rand(3, 320, 320, device=RL.DEVICE)])
    settle()
    time.sleep(1.0)

    base_used = sample()
    base_rss = rss_mb()
    base_cuda = cuda_mb()

    # 2) Load the estimator for this condition (fresh imports + weights).
    keep = None
    if cond == "detector_only":
        pass  # noise floor: load nothing
    elif cond == "iounet_gpu":
        keep = RL._load_mlp_head(RL.IoUPredictor, RL.IOU_CKPT)
    elif cond == "gfl_gpu":
        keep = RL._load_mlp_head(RL.GFLPredictor, RL.GFL_CKPT)
    elif cond == "ptc_cpu":
        keep = RL.PTCEstimator(conf_thr=RL.PTC_CONF_THR)
    elif cond == "xgboost_cpu":
        keep = RL.load_xgboost()  # imports sklearn/xgboost/scipy + unpickles model
    else:
        raise SystemExit(f"unknown cond {cond}")

    settle()
    time.sleep(1.0)

    end_used = sample()
    end_rss = rss_mb()
    end_cuda = cuda_mb()

    _ = keep  # keep the estimator alive through the measurement
    return {
        "used_ram_mb": round(end_used - base_used, 1),
        "rss_mb": round(end_rss - base_rss, 1),
        "cuda_mb": round(end_cuda - base_cuda, 1),
        "base_used_mb": round(base_used, 1),
        "end_used_mb": round(end_used, 1),
    }


# ── parent: orchestrate one fresh process per condition ───────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS, help="measure a single condition (child mode)")
    ap.add_argument("--repeats", type=int, default=2,
                    help="fresh-process repeats per condition (parent mode)")
    ap.add_argument("--only", choices=CONDS, help="measure just one condition, merge into existing json")
    args = ap.parse_args()

    if args.cond:
        print("RAMJSON:" + json.dumps(measure_one(args.cond)))
        return

    conds = [args.only] if args.only else CONDS
    dst = RESULTS / "ram_overhead.json"
    out = json.loads(dst.read_text()) if (args.only and dst.exists()) else {}
    for cond in conds:
        runs = []
        for r in range(args.repeats):
            print(f"[{cond}] repeat {r + 1}/{args.repeats} ...", flush=True)
            p = subprocess.run(
                [sys.executable, __file__, "--cond", cond],
                capture_output=True, text=True, env={**os.environ},
            )
            line = next((l for l in p.stdout.splitlines() if l.startswith("RAMJSON:")), None)
            if line is None:
                print("  !! child failed:")
                print(p.stdout[-1200:]); print(p.stderr[-1200:])
                continue
            runs.append(json.loads(line[len("RAMJSON:"):]))
        if not runs:
            print(f"  → {cond}: NO successful runs, skipping", flush=True)
            continue
        # headline = RSS delta (unified memory: captures GPU weights + CPU libs;
        # immune to page-cache/background noise unlike system used-RAM).
        runs.sort(key=lambda d: d["rss_mb"])
        med = dict(runs[len(runs) // 2])
        med["all_runs"] = runs
        out[cond] = med
        print(f"  → {cond}: RSS Δ={med['rss_mb']}MB  "
              f"(cuda Δ={med['cuda_mb']}, sys-used Δ={med['used_ram_mb']})", flush=True)
        dst.write_text(json.dumps(out, indent=2))  # save incrementally

    print(f"\nSaved: {dst}")
    print("Paper Fig3(a) RAM overhead:  IoU +68  GFL +76  PTC +63  XGB +141 MB")
    print("(headline = RSS Δ; note PTC is parameter-free so its true footprint ≈ 0,")
    print(" exposing the whole-system RAM noise in the paper's +63 for PTC.)")


if __name__ == "__main__":
    main()
