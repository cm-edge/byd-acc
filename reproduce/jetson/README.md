# reproduce/jetson — figure scripts and machine-state tools

Entry point for reproducing the Jetson figures is the step-by-step manual
[`REPRODUCING.md`](../../REPRODUCING.md) (repo root). This directory holds the
scripts that manual drives.

## Figure scripts

| Script | Paper artifact | Input |
|---|---|---|
| `fig2.py` | Fig 2: per-frame latency + throughput | `jetson/benchmark/results/latency.json`, `throughput.json` |
| `fig3.py` | Fig 3: resource deltas, GPU utilisation, power/energy | `resources.json`, `resources_throughput.json`, `energy.json` |
| `fig4.py` | Appendix B Fig 4: motivation study | `motivation_results.json` |

The input JSONs are produced by the benchmark runs (REPRODUCING.md steps 5-8)
and are not tracked by git.

## Machine-state / measurement tools

| Script | Purpose |
|---|---|
| `bg_load.py` | Calibrated background load; also the no-root CPU clock-ramp spinner (REPRODUCING §1.2/§6) |
| `churn_load.py` | Fragmented preemption load, used by `run_paper_state_bench.sh` (configs Aprime/App) |
| `measure_ram_overhead.py` | Deterministic per-estimator RAM footprint (fresh process, RSS delta) — alternative to the noisy whole-system RAM deltas (REPRODUCING §10) |
| `run_paper_state_bench.sh` | Machine-state emulation driver: rebuilds the paper-era 1344 MHz busy-clock state (no root) and runs the shipped latency/throughput benchmark, logging the achieved CPU frequencies (REPRODUCING §6) |

Note: the benchmarks overwrite `jetson/benchmark/results/` in place. If you
want to keep a run, copy its JSONs somewhere before the next run.
