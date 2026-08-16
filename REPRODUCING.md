# Reproducing the Jetson results (Fig. 2 / 3 / 4) — step-by-step manual

This manual reproduces the system-performance figures of the paper on an NVIDIA
Jetson Orin Nano. The benchmark scripts write their result JSONs to
`jetson/benchmark/results/` (not tracked by git); each figure comes from one
complete, uninterrupted execution of the corresponding script. The expected
values quoted below are from our own reference runs on the same board.

Edge-device benchmarks are sensitive to machine state. Steps 1-2 pin that state
down. If you follow them, every quantity lands within the tolerances in section 10.
All states are reachable **without root** (except the one-time power-mode
setting), using only tools in this repository.

---

## 1. Host setup (one-time)

**1.1 Power mode — MAXN:**

```bash
sudo nvpmodel -m 2        # MAXN on the Orin Nano developer kit
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq_dev/max_freq   # expect 1020000000
```

**1.2 CPU clocks — pin to maximum for latency measurements.** Surviving
telemetry from the paper-era sessions (tegrastats, 2026-05-28) shows the busy
cores sat in the 1344 MHz band. They were raised there by desktop background
activity through schedutil, not pinned at 1728 MHz. Per-frame latency has a
host-side part (kernel-launch dispatch, pre/post-processing) of about 15-20 ms,
and that part scales with the CPU clock. On today's machine the detector floor
reads about 136.6 ms with clocks pinned at 1728 MHz, about 138-139.6 ms in
emulated 1344 MHz-band states, and about 143 ms on a quiet machine under stock
DVFS. So there are two recipes. Pinning to 1728 MHz (below) reproduces the
paper's floor and the PTC-IoU overhead most closely. The machine-state
emulation driver `reproduce/jetson/run_paper_state_bench.sh` (step 6)
reproduces the whole Fig 2(a) column in one state. To pin the clocks
(one-time until reboot; **cpuidle is untouched**, unlike `jetson_clocks`):

```bash
echo 1728000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq \
                        /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
# restore stock DVFS afterwards:
echo 729600  | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq \
                        /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
```

No-root alternative: the full-duty spinner recipe in step 6 reaches the same
1728 MHz state (verified frequency-equivalent). The cost is that it occupies
one core per cpufreq policy. Either way, verify the surrounding state:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor       # schedutil
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq       # 1728000
cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable         # 0  (must stay 0)
```

> ⚠ **Never run `jetson_clocks`.** It pins the clocks **and disables cpuidle**.
> That changes thread-synchronisation behaviour: sequential-vs-pipelined
> orderings flip, measured GPU utilisation reads too high, and measured CPU%
> reads too low. If it was ever run, reboot. The `scaling_min_freq` method
> above pins clocks without this side effect.

**1.3 Host packages.** Energy monitoring needs jetson-stats, and the figure
scripts (step 9) plus `run_resources.py` need numpy and matplotlib on the host:

```bash
sudo pip3 install jetson-stats && sudo systemctl start jtop.service
pip3 install numpy matplotlib
```

**1.4 Keep the host quiet.** Close editors and browsers. Do not run anything
during a measurement except what a recipe lists. Even the recipe's own monitors
cost about +1.8 ms/frame (that is why the Fig 2a recipe runs without them).

## 2. Container setup (one-time)

Use `dustynv/jetson-inference:r36.3.0` (torch 2.2 + torchvision + CUDA for sm_87).
Host-pip torch will not see the GPU, and `dustynv/pytorch:2.1-r36.2.0` lacks
torchvision.

```bash
docker pull dustynv/jetson-inference:r36.3.0
docker run -d --name byd-jetson --runtime nvidia \
    -v /path/to/byd-acc:/workspace \
    -e TORCH_HOME=/workspace/data/torch_hub \
    dustynv/jetson-inference:r36.3.0 sleep infinity
docker exec byd-jetson pip3 install --index-url https://pypi.org/simple \
    scipy==1.15.3 joblib==1.5.3 threadpoolctl==3.6.0 \
    scikit-learn==1.7.2 xgboost==3.2.0
```

> The `--index-url` override is **required**. The image ships with a `PIP_INDEX_URL`
> pointing at a defunct NVIDIA mirror (`jetson.webredirect.org`), so a bare
> `pip3 install` never reaches PyPI and fails with "No matching distribution found".

> If `data/MOT17` or `data/torch_hub` under your checkout are **symlinks** (e.g. into
> a shared dataset directory), bind-mount the link **targets** as well
> (`-v /path/to/target:/path/to/target:ro`). Otherwise the links dangle inside the
> container and step 4 fails with "No such file or directory".

Keep the container's numpy 1.26.x. scikit-learn must be **1.7.2** (the version
`weights/xgboost/predictor.pkl` was saved under). See `jetson/README.md` for details.

## 3. Data (one-time)

- **MOT17-02-FRCNN frames** under `data/MOT17/train/MOT17-02-FRCNN/img1/`
  (from the official MOT17 release, motchallenge.net).
- **Detector weights** cached under `data/torch_hub/` — populate once with internet:
  `docker exec -w /workspace byd-jetson python3 -c "import torchvision;
  torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT');
  torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')"`.
  (`TORCH_HOME` must stay pointed here: a mid-benchmark re-download can abort with a
  heap corruption on Jetson.)

## 4. Sanity check (~3 min)

```bash
docker exec -w /workspace byd-jetson \
    python3 jetson/benchmark/run_latency.py --conditions detector_only
```

Expect a detector floor of **~136-137 ms** with the CPU clocks pinned per step 1.2,
or **~143 ms** in the stock idle-DVFS state (the +7 ms is the host-side share of
the frame running at lower CPU clocks). If you see ~189 ms, the GPU is capped
(power mode, step 1.1).

## 5. Fig 2(b) + Fig 3(b) — throughput and GPU utilisation (~10 min)

One session. The background load calibrates the desktop-session CPU/RAM level the
paper was measured under (`reproduce/jetson/bg_load.py` ships in this repo):

```bash
# terminal 1 (host): background load, leave running
setsid python3 reproduce/jetson/bg_load.py --cores 0.5 --duty 1.0 --mb 1500 &
# terminal 2 (host): GPU/CPU monitor — waits for the benchmark's timing markers
python3 jetson/benchmark/run_resources.py --mode throughput
# terminal 3 (container): the benchmark (all 6 conditions)
docker exec -w /workspace byd-jetson python3 jetson/benchmark/run_throughput.py
# afterwards, stop the background load:
pkill -f bg_load.py
```

Produces `throughput.json` + `resources_throughput.json`. Expected:
IoU-Net 6.5 FPS (1.00×), PTC-IoU 1.06-1.08× seq / pipe slightly below seq,
XGBoost ≈0.88-0.90× seq, GPU utilisation ≈85-88 % for the heads and PTC, with the
XGBoost starvation dip clearly below them (seq ≈70 %, pipe can read several
points higher).

## 6. Fig 2(a) — per-frame latency (~8 min, NO host monitors)

**If you pinned the clocks per step 1.2, no spinner or core-pinning is needed.**
Run the benchmark directly:

```bash
docker exec -w /workspace byd-jetson python3 jetson/benchmark/run_latency.py
```

**No-root alternative** (frequency-equivalent, verified): a full-duty spinner per
cpufreq policy ramps the cores to 1728 MHz while cpuidle stays enabled; pin the
benchmark to the remaining cores:

```bash
setsid taskset -c 3 python3 reproduce/jetson/bg_load.py --cores 1 --duty 1.0 --mb 1500 &
docker exec -w /workspace byd-jetson taskset -c 0-2 python3 jetson/benchmark/run_latency.py
# afterwards, stop the spinner:
pkill -f bg_load.py
```

**In both cases, save this run's output now.** Steps 7 and 8 rerun
run_latency.py and overwrite `latency.json` (step 8 even leaves a 2-condition
file), while fig2.py reads `latency.json`. Step 9 restores this copy:

```bash
cp jetson/benchmark/results/latency.json jetson/benchmark/results/latency.fig2a.json
```

Run **nothing else on the host**, in particular not the step-5/7 monitors
(≈ +1.8 ms/frame). Produces `latency.json`. Expected: detector
≈137 ms, +IoU-Net/+GFL ≈ +10 ms, +PTC-IoU ≈ +0.5 ms, +XGBoost ≈ +27 ms.

XGBoost note: pinned to 3 coherent 1728 MHz cores its estimator step runs ≈26 ms.
On 6 mixed-clock DVFS cores the same step measures ≈31 ms, and that is where the
paper's higher XGBoost E2E (168.5 ms) comes from. Both values are real; the
result depends on which cores the thread pool lands on.

**Single-state alternative (whole Fig 2(a) column at once, no root):** the
paper-era busy-clock state (1344 MHz band, step 1.2) can be rebuilt with calibrated
loads, running the shipped benchmark inside it:

```bash
reproduce/jetson/run_paper_state_bench.sh latency App 1
```

Expected (two verified rounds): floor 138.1-138.3 ms (+1.6-1.8 %
vs paper), IoU-Net/GFL ≈ +10.3-10.8 ms, PTC-IoU +0.8-1.2 ms, XGBoost
+32.2-34.0 ms (paper +32.6). The three load configurations (A, Aprime, App)
are described in the header of `run_paper_state_bench.sh`; the script also
logs the achieved CPU frequencies of every run as evidence of the state.

## 7. Fig 3(a) + Fig 3(c) — resource deltas, power/energy (~10 min)

Same machine state as step 6, plus the two host monitors, plus
`LAZY_XGB_IMPORT=1`. The monitors cost latency, which is why this session's
`latency.json` is *not* used for Fig 2a. `LAZY_XGB_IMPORT=1` charges the
XGBoost stack's import footprint to that condition's RAM window; this is the
accounting behind the paper's +141 MB. (The default pre-import, a Jetson
stability workaround, would book the imports into the shared baseline instead.)

```bash
setsid taskset -c 3 python3 reproduce/jetson/bg_load.py --cores 1 --duty 1.0 --mb 1500 &
python3 jetson/benchmark/run_resources.py --mode latency     # terminal 2 (host)
python3 jetson/benchmark/run_energy.py                       # terminal 3 (host)
docker exec -w /workspace -e LAZY_XGB_IMPORT=1 byd-jetson taskset -c 0-2 \
    python3 jetson/benchmark/run_latency.py                  # terminal 4
# afterwards, stop the spinner (a leftover full-duty spinner would pollute step 8):
pkill -f bg_load.py
```

Produces `resources.json` + `energy.json` (this session's `latency.json` is
monitored — do **not** let it replace the step-6 file). Expected:
CPU deltas negative for the three heads; RAM deltas: heads ≈ +50-90 MB, XGBoost
clearly the largest (≈ +90-150 MB), with individual values inside the ±60 MB noise
floor of section 10; power ordering XGBoost < detector < heads < PTC-IoU within
~5 %; energy overhead **negative for PTC-IoU only** and largest for XGBoost.

## 8. Fig 4 — motivation study (~6 min)

```bash
setsid python3 reproduce/jetson/bg_load.py --cores 0.08 --mb 0 &
python3 jetson/benchmark/run_resources.py --mode motivation  # terminal 2 (host)
docker exec -w /workspace byd-jetson python3 jetson/benchmark/run_latency.py \
    --conditions detector_only gfl_gpu                       # terminal 3
# afterwards:
pkill -f bg_load.py
```

Before starting, make sure no full-duty spinner from step 6/7 is still running
(`pgrep -af bg_load.py`) — it would inflate this figure's CPU numbers.

Produces `motivation_results.json`. Expected: GPU ≈81 % rising to ≈85 % under
co-location; CPU ≈13-14 % and *not* rising.

## 9. Generate the figures

```bash
# restore the step-6 latency data first (steps 7/8 overwrote latency.json):
cp jetson/benchmark/results/latency.fig2a.json jetson/benchmark/results/latency.json
python3 reproduce/jetson/fig2.py --plot-only
python3 reproduce/jetson/fig3.py --plot-only
python3 reproduce/jetson/fig4.py --plot-only
```

Outputs land in `reproduce/output/`.

## 10. Expected tolerances (measured run-to-run variance on our board)

| Quantity | Tolerance | Notes |
|---|---|---|
| Per-frame latency (mean) | ±2 % within a clock state | ≈ +5 % without the step-6 ramped-clock recipe |
| Estimator overhead (ms vs floor) | ±1 ms within a clock state | across clock states the GPU-head overhead itself moves by ≈2 ms (step 6 note) |
| Throughput FPS | ±1 % within a state | ratios vs IoU-Net: ±0.02 |
| GPU utilisation | ±5 pts | per-sample std is 20-35; faster host dispatch reads higher |
| Board power (VDD_IN) | ±5 % | ordering (XGBoost lowest, PTC-IoU highest) is stable |
| Energy overhead % | ±3 pts | signs are stable: PTC-IoU negative, XGBoost worst |
| CPU Δ vs detector-only | sign stable, ±1.5 pts | |
| Whole-system RAM Δ | **±60 MB — treat as noise** | deterministic alternative: process-RSS via `reproduce/jetson/measure_ram_overhead.py` (IoU-Net +58 / GFL +58 / PTC-IoU +0 / XGBoost +121 MB) |

## 11. Known high-variance points

- **Pipelined-XGBoost throughput (paper: 0.89×).** The producer/consumer overlap of
  the pipelined runner is governed by Python GIL arbitration. It measures 0.94-0.97×
  across 10+ re-runs on our board, over sweeps of core count, background load, CPU
  min-frequency and cpuidle state. The paper's 0.89× is the low end of that
  behaviour, consistent with its own remark that pipelining gains are "largely
  cancelled by thread-synchronisation overhead". The conclusion — pipelined XGBoost
  stays below 1.0× — holds at any point in the range. Its Fig 3(b) GPU utilisation
  (70.7 %) moves with the same mechanism. Preemption-load experiments
  (`run_paper_state_bench.sh`, config Aprime) move *sequential* XGBoost onto
  0.88-0.89× (paper 0.87×) but push the *pipelined* ratio above 1.0 (1.00-1.01×):
  added scheduler churn increases the share of consumer work masked inside the GPU
  window. The exact 0.89× therefore belongs to the paper-era machine state, which
  no longer exists on this board.
- **Whole-system RAM deltas** (±60 MB noise floor): compare signs and magnitudes,
  not exact values; use the RSS tool for deterministic per-estimator footprints.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `corrupted size vs. prev_size` abort | Known Jetson glibc heap issue. At startup or with `LAZY_XGB_IMPORT=1`: just retry the run. (The scripts pre-import the XGBoost stack before torch by default, which prevents the mid-run variant.) It can strike several runs in a row — keep retrying; the container itself is fine (verify with a bare `docker exec byd-jetson python3 -c "import pandas, scipy, sklearn, xgboost, torch"`). |
| After a benchmark abort, the host monitor sits at "Waiting for … markers" forever | `run_resources.py` / `run_energy.py` have no timeout — a crashed benchmark never writes its timing markers. Ctrl-C the monitor(s), kill any leftover `tegrastats`, then restart the session: monitors first, benchmark second. |
| Detector floor ≈189 ms | GPU capped at 612 MHz — set MAXN (step 1.1). |
| Detector floor ≈143 ms in step 6 | Spinner not on the benchmark's cpufreq policy (must be cpu3 with benchmark on cpu0-2), or `taskset` missing. |
| Latency ~2 ms above reference | A host monitor (tegrastats/jtop) or other process ran during the measurement. |
| `InconsistentVersionWarning` on predictor.pkl | scikit-learn ≠ 1.7.2 — reinstall (step 2). |
| pip: `No matching distribution found` in container | The image's built-in `PIP_INDEX_URL` points at a dead mirror — add `--index-url https://pypi.org/simple` (step 2). |
| Container can't see `data/MOT17` though the host path exists | `data/` entries are symlinks — bind-mount the link targets too (step 2 note). |
| Fig 2(a) wrong or fig2.py fails after steps 7/8 | `latency.json` was overwritten by a later step — restore the step-6 copy (steps 6/9). |
| Fig 4 CPU% reads ≈15 pts high | A full-duty spinner from step 6/7 was still running — `pkill -f bg_load.py` first (step 8). |
| CUDA "no kernel image" | Host-pip torch shadowing the container's — don't set `PYTHONPATH` to host site-packages. |
| Benchmark re-downloads detector weights, then aborts | `TORCH_HOME` not pointing at `/workspace/data/torch_hub` (step 2/3). |
