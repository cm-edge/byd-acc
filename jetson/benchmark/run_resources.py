#!/usr/bin/env python3
"""
Resource monitor — run on the HOST while run_latency.py runs in the container.

Collects CPU / GPU / RAM metrics via tegrastats for each of the 5 conditions:
  detector_only, iounet_gpu, gfl_gpu, ptc_cpu, xgboost_cpu

Usage:
  # Terminal 1 (host):
      python3 motivation_study/run_resources.py

  # Terminal 2 (container):
      python3 run_latency.py

run_latency.py writes wall-clock timestamps to results/timing.json as it
progresses.  This script polls that file, then slices the tegrastats log
by condition and aggregates mean/std per metric.

Results saved to motivation_study/results/resources.json.
"""

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

RESULTS_DIR  = Path(__file__).parent / "results"
TEGRASTATS   = "/usr/bin/tegrastats"
LOG_FILE     = RESULTS_DIR / "tegrastats_resources.log"
TIMING_FILE  = RESULTS_DIR / "timing.json"
OUT_FILE     = RESULTS_DIR / "resources.json"

# Conditions for run_latency.py (sequential only)
CONDITIONS_LATENCY = [
    "detector_only",
    "iounet_gpu",
    "gfl_gpu",
    "ptc_cpu",
    "xgboost_cpu",
]

# Conditions for run_throughput.py (includes seq + pipelined)
CONDITIONS_THROUGHPUT = [
    "iounet_gpu",
    "gfl_gpu",
    "ptc_cpu_seq",
    "ptc_cpu_pipelined",
    "xgboost_cpu_seq",
    "xgboost_cpu_pipelined",
]

# tegrastats line patterns (Jetson JetPack 4.x / 5.x compatible)
_GPU_RE  = re.compile(r'GR3D_FREQ\s+(\d+)%')
_RAM_RE  = re.compile(r'RAM\s+(\d+)/(\d+)MB')
_CPU_RE  = re.compile(r'CPU\s+\[([^\]]+)\]')
_SWAP_RE = re.compile(r'SWAP\s+(\d+)/\d+MB')


def parse_line(line: str) -> dict:
    s = {}
    m = _GPU_RE.search(line)
    if m:
        s['gpu_pct'] = float(m.group(1))

    m = _RAM_RE.search(line)
    if m:
        s['ram_used_mb']  = float(m.group(1))
        s['ram_total_mb'] = float(m.group(2))
        s['ram_pct']      = round(100.0 * float(m.group(1)) / float(m.group(2)), 1)

    m = _CPU_RE.search(line)
    if m:
        utils = [float(p.split('%')[0]) for p in m.group(1).split(',') if '%' in p]
        if utils:
            s['cpu_pct_mean'] = round(sum(utils) / len(utils), 1)
            s['cpu_pct_max']  = max(utils)

    m = _SWAP_RE.search(line)
    if m:
        s['swap_used_mb'] = float(m.group(1))

    return s


def aggregate(samples: list[dict]) -> dict:
    if not samples:
        return {}
    import numpy as np
    keys = set().union(*samples)
    out = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if vals:
            a = np.array(vals, dtype=np.float64)
            out[f'{k}_mean'] = round(float(np.mean(a)), 2)
            out[f'{k}_std']  = round(float(np.std(a)), 2)
            out[f'{k}_p5']   = round(float(np.percentile(a,  5)), 2)
            out[f'{k}_p25']  = round(float(np.percentile(a, 25)), 2)
            out[f'{k}_p75']  = round(float(np.percentile(a, 75)), 2)
            out[f'{k}_p95']  = round(float(np.percentile(a, 95)), 2)
    return out


def wait_for_all_markers(conditions: list, timing_file: Path,
                         label: str, poll: float = 0.5) -> dict:
    needed = {f"{c}_start" for c in conditions} | {f"{c}_end" for c in conditions}
    seen   = set()
    print(f"Waiting for {label} ({len(needed)} markers) ...")
    while True:
        if timing_file.exists():
            try:
                markers = json.loads(timing_file.read_text())
            except json.JSONDecodeError:
                time.sleep(poll)
                continue
            for k in (needed & set(markers.keys())) - seen:
                print(f"  ✓ {k}")
                seen.add(k)
            if needed.issubset(markers.keys()):
                print("All markers received.\n")
                return markers
        time.sleep(poll)


def slice_log(entries: list[tuple], t_start: float, t_end: float) -> list[dict]:
    return [s for ts, s in entries if t_start <= ts <= t_end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=200,
                        help="tegrastats sampling interval in ms (default 200)")
    parser.add_argument("--no-tegrastats", action="store_true",
                        help="skip tegrastats (parse existing log file)")
    parser.add_argument("--mode", choices=["latency", "throughput"], default="latency",
                        help="which benchmark to monitor (default: latency)")
    args = parser.parse_args()

    # Select conditions and timing file based on mode
    if args.mode == "throughput":
        CONDITIONS  = CONDITIONS_THROUGHPUT
        TIMING_FILE = RESULTS_DIR / "timing_throughput.json"
        OUT_FILE    = RESULTS_DIR / "resources_throughput.json"
        label       = "run_throughput.py"
    else:
        CONDITIONS  = CONDITIONS_LATENCY
        TIMING_FILE = RESULTS_DIR / "timing.json"
        OUT_FILE    = RESULTS_DIR / "resources.json"
        label       = "run_latency.py"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_entries: list[tuple[float, dict]] = []

    if args.no_tegrastats:
        # Re-parse an existing log file (useful for re-running analysis)
        print(f"Re-parsing existing log: {LOG_FILE}")
        if not LOG_FILE.exists():
            print(f"ERROR: {LOG_FILE} not found.")
            return
        with LOG_FILE.open() as f:
            for line in f:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    try:
                        ts = float(parts[0])
                        s  = parse_line(parts[1])
                        if s:
                            log_entries.append((ts, s))
                    except ValueError:
                        pass
        print(f"Loaded {len(log_entries)} samples from existing log.")
        markers = json.loads(TIMING_FILE.read_text())
    else:
        LOG_FILE.unlink(missing_ok=True)
        TIMING_FILE.unlink(missing_ok=True)

        print(f"Starting tegrastats  (interval={args.interval}ms)")
        print(f"Switch to container and run:  python3 {label}\n")

        proc = subprocess.Popen(
            [TEGRASTATS, "--interval", str(args.interval)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )

        def _reader():
            with LOG_FILE.open("a") as f:
                for line in proc.stdout:
                    ts = time.time()
                    f.write(f"{ts:.3f} {line}")
                    f.flush()
                    s = parse_line(line)
                    if s:
                        log_entries.append((ts, s))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        try:
            markers = wait_for_all_markers(CONDITIONS, TIMING_FILE, label)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            proc.terminate()
            return

        proc.terminate()
        proc.wait(timeout=3)
        t.join(timeout=2)
        print(f"Collected {len(log_entries)} tegrastats samples total.\n")

    # ── Slice and aggregate per condition ─────────────────────────────────────
    results = {}
    for cond in CONDITIONS:
        t0 = markers.get(f"{cond}_start")
        t1 = markers.get(f"{cond}_end")
        if t0 is None or t1 is None:
            print(f"  WARNING: missing markers for {cond}, skipping.")
            continue
        samples = slice_log(log_entries, t0, t1)
        agg     = aggregate(samples)
        agg["duration_s"] = round(t1 - t0, 2)
        agg["n_samples"]  = len(samples)
        results[cond] = agg

        dur = t1 - t0
        gpu  = agg.get("gpu_pct_mean",      "—")
        cpu  = agg.get("cpu_pct_mean_mean",  "—")
        ram  = agg.get("ram_used_mb_mean",   "—")
        print(f"  {cond:<20s}  {dur:5.1f}s  "
              f"GPU={gpu}%  CPU={cpu}%  RAM={ram}MB  "
              f"({len(samples)} samples)")

    # ── Merge with latency/throughput results ────────────────────────────────
    perf_file = RESULTS_DIR / ("throughput.json" if args.mode == "throughput" else "latency.json")
    if perf_file.exists():
        perf = json.loads(perf_file.read_text())
        for cond in CONDITIONS:
            if cond in perf and cond in results:
                if args.mode == "throughput":
                    results[cond]["fps"]           = perf[cond].get("fps")
                    results[cond]["frame_ms_mean"] = perf[cond].get("frame_ms_mean")
                else:
                    results[cond]["latency_mean_ms"] = perf[cond].get("mean_ms")
                    results[cond]["latency_p95_ms"]  = perf[cond].get("p95_ms")

    out_path = OUT_FILE
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    cols = ["gpu_pct_mean", "cpu_pct_mean_mean", "ram_used_mb_mean"]
    hdrs = ["GPU %", "CPU %", "RAM MB"]
    w = 20
    print(f"\n{'Condition':<{w}}  {'GPU %':>7}  {'CPU %':>7}  {'RAM MB':>9}")
    print("-" * (w + 30))
    for cond in CONDITIONS:
        r = results.get(cond, {})
        row = [f"{r.get(c, float('nan')):>7.1f}" if isinstance(r.get(c), float) else f"{'—':>7}"
               for c in cols]
        print(f"  {cond:<{w-2}}  {'  '.join(row)}")


if __name__ == "__main__":
    main()
