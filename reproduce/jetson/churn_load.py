#!/usr/bin/env python3
"""Fragmented preemption load: N processes, each waking every `period` ms and
busy-spinning for `busy` ms. Defaults reproduce the load shape used in the
2026-07-15 pipe-stall investigation (10 processes, ~2000 wakeups/s total),
emulating the scheduling footprint of the paper-era desktop session
(vscode-server, browser, monitors)."""
import argparse, multiprocessing as mp, time

def worker(period_s, busy_s):
    while True:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < busy_s:
            pass
        time.sleep(max(0.0, period_s - busy_s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--period-ms", type=float, default=5.0)
    ap.add_argument("--busy-ms", type=float, default=0.7)
    a = ap.parse_args()
    procs = []
    for _ in range(a.procs):
        p = mp.Process(target=worker,
                       args=(a.period_ms / 1e3, a.busy_ms / 1e3), daemon=True)
        p.start()
        procs.append(p)
    print(f"[churn_load] {a.procs} procs, period {a.period_ms} ms, "
          f"busy {a.busy_ms} ms (~{a.procs * 1000 / a.period_ms:.0f} wakeups/s)",
          flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
