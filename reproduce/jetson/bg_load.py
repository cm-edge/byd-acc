#!/usr/bin/env python3
"""Stable, calibrated background load to reproduce the paper's measurement machine state.

The paper's measurements were taken with a desktop session active on the host
(~20.8% system CPU, ~5991 MB RAM used -- not a perfectly clean machine). This tool
adds a CONSTANT, documented background load so re-measurements match those conditions:
  - CPU: `cores` worker processes each busy `duty` fraction of the time (multiprocessing,
    so they actually occupy separate cores despite the GIL).
  - RAM: allocate `mb` MB and touch every page so it stays resident.

Usage (run on HOST, in background, during the Fig3 measurement):
    python3 bg_load.py --cores 1.3 --duty 1.0 --mb 800
    # → ~1.3 cores busy ≈ +21% system CPU on a 6-core box; +800 MB RAM
Ctrl-C or kill to stop.
"""
import argparse, multiprocessing as mp, time, sys


def _burn(duty):
    # busy `duty` fraction of each 50 ms slice; idle the rest
    period = 0.05
    while True:
        t = time.perf_counter()
        while time.perf_counter() - t < duty * period:
            pass
        rest = (1.0 - duty) * period
        if rest > 0:
            time.sleep(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=float, default=1.3, help="number of cores to occupy (fractional ok)")
    ap.add_argument("--duty", type=float, default=1.0, help="busy fraction per worker (0-1)")
    ap.add_argument("--mb", type=int, default=800, help="MB of RAM to allocate and keep resident")
    a = ap.parse_args()

    # RAM: allocate + touch pages
    blob = None
    if a.mb > 0:
        blob = bytearray(a.mb * 1024 * 1024)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        print(f"[bg_load] allocated {a.mb} MB", flush=True)

    # CPU: full workers + one fractional worker
    n_full = int(a.cores)
    frac = a.cores - n_full
    procs = []
    for _ in range(n_full):
        p = mp.Process(target=_burn, args=(a.duty,), daemon=True); p.start(); procs.append(p)
    if frac > 0:
        p = mp.Process(target=_burn, args=(a.duty * frac,), daemon=True); p.start(); procs.append(p)
    print(f"[bg_load] {a.cores} cores @ duty {a.duty}  ({len(procs)} workers). Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
