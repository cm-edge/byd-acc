# Jetson Benchmark

System-performance measurements for the paper (Figs 2, 3, Appendix B Fig 4).
Must be run on **NVIDIA Jetson Orin Nano** with JetPack 6.x.

## Environment

The benchmark runs inside a Docker container that provides PyTorch + torchvision
for aarch64.  Resource and energy measurements are collected on the HOST using
`tegrastats` and `jetson-stats` (jtop).

### Build / pull the container

```bash
# Option A — use the official Dusty-NV container (recommended):
docker pull dustynv/pytorch:2.1-r36.2.0

# Option B — build from the provided Dockerfile (if customised):
docker build -t byd-acc-jetson .
```

### Mount the repo and data into the container

```bash
docker run -it --rm --runtime nvidia \
    -v /path/to/byd-acc:/workspace \
    -v /path/to/data:/workspace/data \
    -v /path/to/byd-acc/jetson/benchmark/results:/workspace/jetson/benchmark/results \
    dustynv/pytorch:2.1-r36.2.0 \
    bash
```

Inside the container, `cd /workspace` and all scripts use relative paths from there.

### Install Python dependencies inside the container

Pre-built aarch64 wheels are in `jetson/wheels/` to avoid re-downloading at runtime:

```bash
pip install jetson/wheels/*.whl
```

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
| `benchmark/plot_tasks.py` | Anywhere | Generate publication-quality figures from result JSONs |

Results are written to `benchmark/results/` (shared between host and container via the volume mount).

## Two-terminal workflow summary

```
Terminal 1 (HOST)                     Terminal 2 (Docker container)
─────────────────────                 ──────────────────────────────
python benchmark/run_resources.py  │  (start after Terminal 1 is ready)
                                   │  python benchmark/run_latency.py
(waits for timing.json, then       │
 slices and saves resources.json)  │
                                   │
python benchmark/run_energy.py     │  python benchmark/run_latency.py
(waits for timing.json, saves      │  (re-run for energy — jtop needs
 energy.json)                      │   separate timing markers)
```

Or use the reproduce/ wrappers which handle the above automatically when
called with the appropriate flags (`--host-monitor` / `--container-bench`).
