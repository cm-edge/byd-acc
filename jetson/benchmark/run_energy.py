#!/usr/bin/env python3
"""
Energy profiling — run on the HOST using jetson-stats (jtop).

Collects per-condition power consumption (mW) and estimates energy (mJ)
for each of the 5 conditions in run_latency.py.

Prerequisites (host):
    sudo pip3 install jetson-stats
    sudo systemctl start jtop.service   # starts the jtop daemon

Usage:
  # Terminal 1 (host):
      python3 jetson/benchmark/run_energy.py

  # Terminal 2 (container):
      python3 run_latency.py

Requires run_latency.py to have already written results/timing.json
(the timing markers it writes during each condition).

Results saved to jetson/benchmark/results/energy.json.
"""

import json
import statistics
import sys
import threading
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
TIMING_FILE = RESULTS_DIR / "timing.json"
OUT_FILE    = RESULTS_DIR / "energy.json"

CONDITIONS = [
    "detector_only",
    "iounet_gpu",
    "gfl_gpu",
    "ptc_cpu",
    "xgboost_cpu",
]


def check_jtop():
    try:
        from jtop import jtop  # noqa: F401
        return True
    except ImportError:
        print("ERROR: jetson-stats not installed.")
        print("  Install with:  sudo pip3 install jetson-stats")
        print("  Then start daemon:  sudo systemctl start jtop.service")
        return False


def collect_power(samples: list, stop_event: threading.Event,
                  interval_s: float = 0.2):
    """Background thread: polls jtop every interval_s and appends samples."""
    from jtop import jtop
    try:
        with jtop(interval=interval_s) as jetson:
            while not stop_event.is_set():
                if jetson.ok():
                    ts  = time.time()
                    pwr = jetson.power

                    # jtop power format varies by JetPack version.
                    # Try tot (total board power), then fallback fields.
                    total_mw = None
                    if isinstance(pwr, dict):
                        # JetPack 5.x: {'rail': {...}, 'tot': {'power': mW, 'curr': mA, ...}}
                        if 'tot' in pwr and isinstance(pwr['tot'], dict):
                            tot = pwr['tot']
                            total_mw = tot.get('power', tot.get('avg', tot.get('curr', None)))
                        # JetPack 4.x: flat dict with 'POM_5V_IN' etc.
                        if total_mw is None:
                            candidates = [v for k, v in pwr.items()
                                          if 'IN' in k or 'total' in k.lower()]
                            if candidates:
                                total_mw = sum(float(c) if isinstance(c, (int, float))
                                               else float(c.get('power', c.get('cur', 0)))
                                               for c in candidates)
                    if total_mw is not None:
                        samples.append((ts, float(total_mw)))
    except Exception as e:
        print(f"  [jtop thread] {e}")


def wait_for_all_markers(poll: float = 0.5) -> dict:
    needed = {f"{c}_start" for c in CONDITIONS} | {f"{c}_end" for c in CONDITIONS}
    seen   = set()
    print(f"Waiting for run_latency.py ({len(needed)} timing markers)...")
    while True:
        if TIMING_FILE.exists():
            try:
                markers = json.loads(TIMING_FILE.read_text())
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


def slice_samples(samples: list[tuple], t0: float, t1: float) -> list[float]:
    return [mw for ts, mw in samples if t0 <= ts <= t1]


def aggregate_power(mw_list: list[float], duration_s: float) -> dict:
    if not mw_list:
        return {}
    mean_mw = statistics.mean(mw_list)
    return {
        "power_mean_mw":  round(mean_mw, 1),
        "power_std_mw":   round(statistics.stdev(mw_list) if len(mw_list) > 1 else 0.0, 1),
        "power_max_mw":   round(max(mw_list), 1),
        "energy_mj":      round(mean_mw * duration_s, 1),  # mW × s = mJ
        "duration_s":     round(duration_s, 2),
        "n_samples":      len(mw_list),
    }


def main():
    if not check_jtop():
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_FILE.unlink(missing_ok=True)

    samples: list[tuple[float, float]] = []   # (wall_time, total_mw)
    stop_ev = threading.Event()

    print("Starting jtop power collection thread...")
    print("Switch to container and run:  python3 run_latency.py\n")

    collector = threading.Thread(
        target=collect_power, args=(samples, stop_ev, 0.2), daemon=True
    )
    collector.start()

    try:
        markers = wait_for_all_markers()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        stop_ev.set()
        collector.join(timeout=3)
        return

    stop_ev.set()
    collector.join(timeout=3)
    print(f"Collected {len(samples)} power samples total.\n")

    # ── Aggregate per condition ───────────────────────────────────────────────
    results = {}
    for cond in CONDITIONS:
        t0  = markers.get(f"{cond}_start")
        t1  = markers.get(f"{cond}_end")
        if t0 is None or t1 is None:
            print(f"  WARNING: missing markers for {cond}, skipping.")
            continue
        dur    = t1 - t0
        mw_lst = slice_samples(samples, t0, t1)
        agg    = aggregate_power(mw_lst, dur)
        results[cond] = agg
        print(f"  {cond:<20s}  {dur:5.1f}s  "
              f"{agg.get('power_mean_mw','—'):>7}mW  "
              f"energy={agg.get('energy_mj','—')}mJ  "
              f"({len(mw_lst)} samples)")

    # ── Normalise energy relative to detector_only ────────────────────────────
    ref_energy = results.get("detector_only", {}).get("energy_mj", None)
    if ref_energy:
        for cond, r in results.items():
            if "energy_mj" in r and ref_energy > 0:
                r["energy_overhead_pct"] = round(
                    100.0 * (r["energy_mj"] - ref_energy) / ref_energy, 1)

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {OUT_FILE}")

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 22
    print(f"\n{'Condition':<{w}}  {'Power mW':>10}  {'Energy mJ':>10}  {'Overhead':>10}")
    print("-" * (w + 38))
    ref_e = results.get("detector_only", {}).get("energy_mj", float("nan"))
    for cond in CONDITIONS:
        r = results.get(cond, {})
        pwr = r.get("power_mean_mw", float("nan"))
        ej  = r.get("energy_mj",     float("nan"))
        ovh = r.get("energy_overhead_pct", 0.0)
        sign = f"+{ovh:.1f}%" if ovh >= 0 else f"{ovh:.1f}%"
        print(f"  {cond:<{w-2}}  {pwr:>10.1f}  {ej:>10.1f}  {sign:>10}")


if __name__ == "__main__":
    main()
