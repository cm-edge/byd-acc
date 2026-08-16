# Jetson Benchmark

System-performance measurements for the paper (Figs 2, 3, Appendix B Fig 4).
Must be run on **NVIDIA Jetson Orin Nano** with JetPack 6.x.

> **Start here:** the step-by-step reproduction manual is [`REPRODUCING.md`](../REPRODUCING.md)
> at the repository root. It covers the machine-state checklist (power mode, CPU
> clocks, background load) that the numbers depend on, the per-figure recipes, and
> the expected values with tolerances.
> This file only documents the environment and the individual scripts.

## Environment

The benchmark runs inside a Docker container that provides CUDA-enabled PyTorch +
torchvision for aarch64 (Orin's sm_87). pip-installed torch on the host will NOT
see the GPU. Resource and energy measurements are collected on the HOST using
`tegrastats` and `jetson-stats` (jtop).

### Pull the container

```bash
# Verified working image (torch 2.2 + torchvision 0.17 + CUDA on Orin):
docker pull dustynv/jetson-inference:r36.3.0
```

Note: `dustynv/pytorch:2.1-r36.2.0` ships without torchvision and will not run
the detector — use the image above.

### Start the container

```bash
docker run -d --name byd-jetson --runtime nvidia \
    -v /path/to/byd-acc:/workspace \
    -e TORCH_HOME=/workspace/data/torch_hub \
    dustynv/jetson-inference:r36.3.0 \
    sleep infinity
```

`TORCH_HOME` must point at the pre-populated cache in the repo. Some base images
override it, and a runtime re-download of the detector weights can abort with a
heap corruption on Jetson. All benchmark commands then run via
`docker exec -w /workspace byd-jetson python3 ...`.

### Install the XGBoost-estimator dependencies inside the container

```bash
docker exec byd-jetson pip3 install --index-url https://pypi.org/simple \
    scipy==1.15.3 joblib==1.5.3 threadpoolctl==3.6.0 \
    scikit-learn==1.7.2 xgboost==3.2.0
```

The `--index-url` override is required. The image ships with a `PIP_INDEX_URL`
pointing at a defunct NVIDIA mirror, so a bare `pip3 install` fails with
"No matching distribution found".

Keep the container's own numpy (1.26.x); upgrading it breaks the bundled torch.
`scikit-learn` must be **1.7.2**: `weights/xgboost/predictor.pkl` was saved under
that version and other versions raise `InconsistentVersionWarning` on load.

### Install energy monitoring on the HOST

```bash
sudo pip3 install jetson-stats
sudo systemctl start jtop.service
```

## Benchmark scripts

| Script | Where to run | What it measures |
|---|---|---|
| `benchmark/run_latency.py` | Docker container | Per-frame end-to-end latency for 5 conditions |
| `benchmark/run_throughput.py` | Docker container | Sustained FPS (sequential + pipelined) |
| `benchmark/run_resources.py` | HOST | GPU/CPU/RAM via tegrastats (sliced by condition) |
| `benchmark/run_energy.py` | HOST | Power draw and energy overhead via jtop |
| `benchmark/estimator_latency_bench.py` | Docker container | Estimator-only incremental latency |
| `benchmark/plot_tasks.py` | Anywhere | Generate publication-style figures from result JSONs |

Useful flags: `run_latency.py --conditions <name...>` and
`run_throughput.py --conditions <name...> [--merge]` run a subset of conditions;
`LAZY_XGB_IMPORT=1` switches the XGBoost stack to lazy (in-condition) imports —
see `REPRODUCING.md` for when each is appropriate.

Results are written to `benchmark/results/` (visible on the host through the
`/workspace` mount). The whole directory is gitignored; the repo ships scripts
and weights only.

## Two-terminal workflow (monitored runs)

The host monitors wait for the timing markers that `run_latency.py` /
`run_throughput.py` write, then slice their own measurements per condition.
Start the monitors first, then the container benchmark; both monitors can watch
the **same** run:

```
Terminal 1..2 (HOST)                       Terminal 3 (Docker container)
─────────────────────────                  ─────────────────────────────
python3 benchmark/run_resources.py      │
python3 benchmark/run_energy.py         │  (start after the monitors)
(both wait for timing.json,             │  python3 benchmark/run_latency.py
 then save resources.json /             │
 energy.json)                           │
```

**Latency numbers intended for Fig 2(a) must instead be measured with NO host
monitors running**: concurrent tegrastats + jtop add ≈ +1.8 ms/frame. See
`REPRODUCING.md` for the full per-figure recipes. The `reproduce/jetson/`
wrappers help with the two-terminal part: `fig3.py` and `fig4.py` take
`--host-monitor` / `--container-bench` / `--plot-only`; `fig2.py` takes
`--plot-only`.
