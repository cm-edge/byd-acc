#!/usr/bin/env python3
"""
E2E latency benchmark — Jetson Orin Nano.

Five conditions (sequential, warmed-up):
  detector_only  — FasterRCNN (GPU) alone
  iounet_gpu     — FasterRCNN (GPU) + IoUNet head (GPU, RoI pool + MLP)
  gfl_gpu        — FasterRCNN (GPU) + GFL head   (GPU, RoI pool + MLP)
  ptc_cpu        — FasterRCNN (GPU) + PTC-IoU     (CPU, no weights)
  xgboost_cpu    — FasterRCNN (GPU) + XGBoost     (CPU, RF gate + regressor)

Timing: wall-clock per frame, WARMUP frames discarded, N_FRAMES measured.
GPU conditions include torch.cuda.synchronize() before/after each call.
Results saved to jetson/benchmark/results/latency.json.
"""

import json
import re
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

# sklearn (in /host-packages) needs real scipy — don't mock scipy at all.
# Just add the host-packages path so all native-install packages are visible.
_HOST_PKGS = "/host-packages"
if _HOST_PKGS not in sys.path:
    sys.path.append(_HOST_PKGS)

import os

# Point torch hub cache to the workspace so the container never downloads
# fasterrcnn_mobilenet_v3_large_fpn at runtime (75 MB download on Jetson
# causes OOM / heap corruption).  Pre-copy the .pth file there once.
_ROOT_EARLY = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME",
                      str(_ROOT_EARLY / "data" / "torch_hub"))

# Import the XGBoost stack BEFORE torch: importing pandas/sklearn after torch
# has been running a frame loop corrupts the glibc heap on Jetson ("corrupted
# size vs. prev_size" abort in the xgboost_cpu condition). Harmless no-op if
# the packages are not installed.
# Set LAZY_XGB_IMPORT=1 to skip this and import lazily inside the xgboost_cpu
# condition instead (the original protocol): whole-system RAM measured for that
# condition then includes the import footprint, matching the paper's Fig 3a
# accounting — at the cost of the occasional heap-corruption abort (just retry).
if os.environ.get("LAZY_XGB_IMPORT") != "1":
    for _mod in ("pandas", "scipy", "sklearn", "xgboost"):
        try:
            __import__(_mod)
        except Exception:
            pass

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "estimator"))

from iounet_inference import (IoUPredictor, _roi_features, ROI_DIM,
                              CONF_THR, CONF_THR_MOT17, MAX_BOXES)
# This benchmark runs on MOT17 (SEQUENCE below). The IoU-Net/GFL GPU head scores
# every box with score >= CONF_THR (up to MAX_BOXES). MOT17's calibrated threshold
# is 0.1 (CONF_THR_MOT17); the module-level CONF_THR default (0.5) is the
# UA-DETRAC/UAVDT value and under-counts RoIs here, shrinking the head cost and
# throwing off Fig 2 vs the paper. Use the dataset-correct threshold.
CONF_THR = CONF_THR_MOT17
from gfl_inference import GFLPredictor
from ptc_inference import PTCEstimator
from feature_extractor import FrameFeatureExtractor, build_window_vector
# DifficultyPredictor is imported lazily inside load_xgboost() to avoid
# pulling sklearn/scipy at module level before the path is fully resolved.

# ── Config ────────────────────────────────────────────────────────────────────
MOT17_ROOT  = ROOT / "data" / "MOT17"
SEQUENCE    = "MOT17-02-FRCNN"
N_FRAMES    = 100
WARMUP      = 20
RESULTS_DIR = Path(__file__).parent / "results"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU_CACHE = 30

IOU_CKPT = ROOT / "weights/neural/iou_net.pth"
GFL_CKPT = ROOT / "weights/neural/gfl.pth"
XGB_CKPT = ROOT / "weights/xgboost/predictor.pkl"

XGB_WINDOW       = 7
IMG_W, IMG_H     = 1920, 1080
VEHICLE_IDS      = {3, 6, 8}   # COCO: car, bus, truck
PTC_CONF_THR     = 0.5

_KEY_MAP = {"fc1": "net.0", "fc2": "net.2", "fc3": "net.4"}


def write_marker(key: str):
    f = RESULTS_DIR / "timing.json"
    markers = json.loads(f.read_text()) if f.exists() else {}
    markers[key] = time.time()
    f.write_text(json.dumps(markers, indent=2))


# ── Resource sampler ──────────────────────────────────────────────────────────

class ResourceSampler:
    """
    Runs tegrastats as a background subprocess (same approach as host_monitor.py
    but from inside the container).  Records timestamped samples for the whole
    benchmark run; each condition's stats are sliced out by start/end time.

    CPU and RAM also read from /proc as a fallback if tegrastats is absent.

    tegrastats output example (old JetPack 4.x rail names; on Orin the rail is VDD_IN and power comes from run_energy.py / run_resources.py):
      RAM 1821/3964MB ... GR3D_FREQ 98% ... CPU [45%@1479,...] ...
      POM_5V_IN 4791/4791 POM_5V_GPU 621/621 POM_5V_CPU 403/403
    """

    TEGRASTATS   = "/usr/bin/tegrastats"
    _GPU_RE  = re.compile(r'GR3D_FREQ\s+(\d+)%')
    _RAM_RE  = re.compile(r'RAM\s+(\d+)/(\d+)MB')
    _CPU_RE  = re.compile(r'CPU\s+\[([^\]]+)\]')
    _PWR_RE  = re.compile(r'POM_5V_IN\s+(\d+)/')   # mW (JetPack 4.x rail name, unused on Orin)

    def __init__(self, interval_ms: int = 200):
        self._interval_ms = interval_ms
        # list of (wall_time, sample_dict) for the entire run
        self._all: list[tuple[float, dict]] = []
        self._proc = None
        self._thread = None
        self._has_tegrastats = Path(self.TEGRASTATS).exists()
        if not self._has_tegrastats:
            print(f"  [ResourceSampler] {self.TEGRASTATS} not found — "
                  "GPU% and Power will be nan; CPU/RAM via /proc.")

    # ── tegrastats line parser ────────────────────────────────────────────────

    @classmethod
    def _parse(cls, line: str) -> dict:
        s = {}
        m = cls._GPU_RE.search(line)
        if m:
            s["gpu_pct"] = float(m.group(1))
        m = cls._RAM_RE.search(line)
        if m:
            s["ram_mb"]  = float(m.group(1))
        m = cls._CPU_RE.search(line)
        if m:
            utils = [float(p.split('%')[0]) for p in m.group(1).split(',') if '%' in p]
            if utils:
                s["cpu_pct"] = sum(utils) / len(utils)
        m = cls._PWR_RE.search(line)
        if m:
            s["power_mw"] = float(m.group(1))
        return s

    # ── /proc fallback ────────────────────────────────────────────────────────

    @staticmethod
    def _proc_cpu() -> float:
        def _read():
            parts = Path("/proc/stat").read_text().splitlines()[0].split()
            v = list(map(int, parts[1:]))
            return v[3] + v[4], sum(v)   # idle+iowait, total
        i0, t0 = _read(); time.sleep(0.05); i1, t1 = _read()
        return 100.0 * (1 - (i1 - i0) / (t1 - t0)) if t1 != t0 else 0.0

    @staticmethod
    def _proc_ram() -> float:
        info = {}
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, v = ln.split(":", 1)
            info[k.strip()] = int(v.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / 1024.0

    # ── background reader ─────────────────────────────────────────────────────

    def _reader(self):
        import subprocess
        if self._has_tegrastats:
            self._proc = subprocess.Popen(
                [self.TEGRASTATS, "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for line in self._proc.stdout:
                ts = time.time()
                s  = self._parse(line)
                if s:
                    self._all.append((ts, s))
        else:
            # fallback: /proc only, no GPU/power
            while not getattr(self, "_stop_fallback", False):
                ts = time.time()
                s  = {"cpu_pct": self._proc_cpu(), "ram_mb": self._proc_ram()}
                self._all.append((ts, s))
                time.sleep(self._interval_ms / 1000.0)

    def start_collecting(self):
        """Start tegrastats for the whole benchmark (call once in main)."""
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop_collecting(self):
        """Stop tegrastats after all conditions are done."""
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._stop_fallback = True
        if self._thread:
            self._thread.join(timeout=3)

    # ── per-condition slice ───────────────────────────────────────────────────

    def get_stats(self, t_start: float, t_end: float) -> dict:
        samples = [s for ts, s in self._all if t_start <= ts <= t_end]
        if not samples:
            return {"n_samples": 0}
        out = {}
        for key in ("cpu_pct", "ram_mb", "gpu_pct", "power_mw"):
            vals = [s[key] for s in samples if key in s]
            if vals:
                out[f"{key}_mean"] = round(statistics.mean(vals), 2)
                out[f"{key}_std"]  = round(
                    statistics.stdev(vals) if len(vals) > 1 else 0.0, 2)
        out["n_samples"] = len(samples)
        return out


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_detector():
    # The weak detector is the full-resolution FasterRCNN-MobileNetV3-L-FPN, the
    # same model the rest of the repo uses (the "-320-" label elsewhere is only a
    # folder name — the underlying model is full-res). The detector variant does
    # not affect the paper's *relative* estimator-overhead conclusions; this is a
    # system-performance benchmark. DETECTOR_RES=320 is available only for
    # exploring how the detector floor shifts, not for reproduction.
    res = os.environ.get("DETECTOR_RES", "full").lower()
    if res == "320":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(DEVICE).eval()
    else:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(DEVICE).eval()
    print(f"  Detector variant: {'320 (min=320)' if res=='320' else 'full-res (min=800)'}")
    return model, weights.transforms()


def _load_mlp_head(cls, ckpt_path):
    head = cls(ROI_DIM).to(DEVICE).eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        mapped = {}
        for k, v in ckpt.items():
            prefix = k.split(".")[0]
            mapped[_KEY_MAP.get(prefix, prefix) + "." + ".".join(k.split(".")[1:])] = v
        head.load_state_dict(mapped)
        print(f"  Loaded {ckpt_path.name}")
    else:
        print(f"  WARNING: {ckpt_path.name} not found — random init (timing only)")
    return head


def load_xgboost():
    from predictor import DifficultyPredictor  # lazy — pulls sklearn/scipy once
    if not XGB_CKPT.exists():
        sys.exit(f"XGBoost model not found: {XGB_CKPT}")
    pred      = DifficultyPredictor.load(XGB_CKPT)
    extractor = FrameFeatureExtractor(image_width=IMG_W, image_height=IMG_H)
    buf       = deque(maxlen=XGB_WINDOW)
    print(f"  Loaded {XGB_CKPT.name}  (window={XGB_WINDOW})")
    return pred, extractor, buf


# ── Frame helpers ─────────────────────────────────────────────────────────────

def get_frames():
    img_dir = MOT17_ROOT / "train" / SEQUENCE / "img1"
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        sys.exit(f"No frames in {img_dir}")
    return frames


def build_gpu_cache(frames, det_tf):
    n = min(N_GPU_CACHE, len(frames))
    print(f"  Preloading {n} frames to GPU...", end=" ", flush=True)
    cache = [det_tf(Image.open(p).convert("RGB")).to(DEVICE) for p in frames[:n]]
    torch.cuda.synchronize()
    print("done.")
    return cache


def _stats(lats):
    a = np.array(lats, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(a)),
        "std_ms":  float(np.std(a)),
        "p5_ms":   float(np.percentile(a,  5)),
        "p25_ms":  float(np.percentile(a, 25)),
        "p50_ms":  float(np.percentile(a, 50)),
        "p75_ms":  float(np.percentile(a, 75)),
        "p95_ms":  float(np.percentile(a, 95)),
        "n":       len(lats),
    }


def _det_to_ptc(det_out):
    """Convert FasterRCNN output → PTC list[(x,y,w,h,conf,cls_str)]."""
    boxes  = det_out["boxes"].cpu().numpy()
    scores = det_out["scores"].cpu().numpy()
    labels = det_out["labels"].cpu().numpy()
    result = []
    for (x1, y1, x2, y2), conf, lbl in zip(boxes, scores, labels):
        if conf >= PTC_CONF_THR and int(lbl) in VEHICLE_IDS:
            result.append((
                float(x1), float(y1),
                float(x2 - x1), float(y2 - y1),
                float(conf), str(int(lbl)),
            ))
    return result


# ── Condition runners ─────────────────────────────────────────────────────────

def run_detector_only(detector, gpu_cache):
    lats  = []
    total = WARMUP + N_FRAMES
    for i in range(total):
        det_t = gpu_cache[i % len(gpu_cache)]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = detector([det_t])
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        if i >= WARMUP:
            lats.append(ms)
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{total}  {ms:.1f} ms")
    return lats


def run_neural_gpu(name, detector, head, gpu_cache):
    """Detector + GPU estimator head.

    Backbone features are captured via a forward hook during the detector's
    own forward pass and reused by _roi_features — the backbone runs only ONCE
    per frame.  Estimator overhead = RoI pool + 3-layer MLP only.
    """
    lats        = []
    total       = WARMUP + N_FRAMES
    _feat_cache = {}
    hook = detector.backbone.register_forward_hook(
        lambda m, inp, out: _feat_cache.__setitem__("features", out)
    )
    try:
        for i in range(total):
            det_t = gpu_cache[i % len(gpu_cache)]
            h, w  = det_t.shape[-2], det_t.shape[-1]

            # Detector pass — hook captures backbone features as a side-effect
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                det_out = detector([det_t])
            torch.cuda.synchronize()
            det_ms = (time.perf_counter() - t0) * 1000

            # GPU estimator: RoI pool + MLP (backbone NOT re-run)
            t1 = time.perf_counter()
            with torch.no_grad():
                boxes  = det_out[0]["boxes"]
                scores = det_out[0]["scores"]
                keep   = scores >= CONF_THR
                boxes  = boxes[keep][:MAX_BOXES]
                if len(boxes) > 0:
                    roi = _roi_features(
                        detector, det_t.unsqueeze(0),
                        [boxes], [(h, w)],
                        cached_features=_feat_cache.get("features"),
                    )
                    _ = head(roi)
            torch.cuda.synchronize()
            est_ms = (time.perf_counter() - t1) * 1000

            ms = det_ms + est_ms
            if i >= WARMUP:
                lats.append(ms)
            if (i + 1) % 40 == 0:
                print(f"    {i+1}/{total}  total={ms:.1f} ms  "
                      f"(det={det_ms:.1f}  est={est_ms:.1f})")
    finally:
        hook.remove()
    return lats


def run_ptc_cpu(detector, ptc_est, gpu_cache):
    """Detector (GPU) + PTC-IoU (CPU). PTC is stateful — reset before run."""
    ptc_est.reset()
    lats  = []
    total = WARMUP + N_FRAMES
    for i in range(total):
        det_t = gpu_cache[i % len(gpu_cache)]

        # Detector (GPU)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            det_out = detector([det_t])
        torch.cuda.synchronize()
        det_ms = (time.perf_counter() - t0) * 1000

        # PTC (CPU) — pure Python/numpy on detection boxes
        t1     = time.perf_counter()
        dets   = _det_to_ptc(det_out[0])
        _      = ptc_est.update(dets)
        est_ms = (time.perf_counter() - t1) * 1000

        ms = det_ms + est_ms
        if i >= WARMUP:
            lats.append(ms)
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{total}  total={ms:.1f} ms  "
                  f"(det={det_ms:.1f}  est_cpu={est_ms:.3f})")
    return lats


def run_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache):
    """Detector (GPU) + XGBoost (CPU). Reset extractor + buffer before run."""
    extractor.reset()
    buf.clear()
    lats  = []
    total = WARMUP + N_FRAMES
    for i in range(total):
        det_t = gpu_cache[i % len(gpu_cache)]

        # Detector (GPU)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            det_out = detector([det_t])
        torch.cuda.synchronize()
        det_ms = (time.perf_counter() - t0) * 1000

        # XGBoost pipeline (CPU):
        #   1. move boxes/scores off GPU
        #   2. extract 28 features
        #   3. predict when window is full (RF gate → XGBoost)
        t1     = time.perf_counter()
        boxes  = det_out[0]["boxes"].cpu().numpy()
        scores = det_out[0]["scores"].cpu().numpy()
        labels = det_out[0]["labels"].cpu().numpy()
        feat   = extractor.update(boxes, scores, labels, VEHICLE_IDS)
        buf.append(feat)
        if len(buf) == XGB_WINDOW:
            X = build_window_vector(list(buf), xgb_pred.feature_columns)
            _ = xgb_pred.predict(X)
        est_ms = (time.perf_counter() - t1) * 1000

        ms = det_ms + est_ms
        if i >= WARMUP:
            lats.append(ms)
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{total}  total={ms:.1f} ms  "
                  f"(det={det_ms:.1f}  est_cpu={est_ms:.3f})")
    return lats


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_CONDITIONS = ["detector_only", "iounet_gpu", "gfl_gpu", "ptc_cpu", "xgboost_cpu"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        choices=ALL_CONDITIONS,
                        help="subset of conditions to measure (default: all). "
                             "The Appendix B Fig 4 motivation study uses "
                             "'--conditions detector_only gfl_gpu'.")
    args = parser.parse_args()
    WANT = set(args.conditions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sequence : {SEQUENCE}")
    print(f"Device   : {DEVICE}")
    print(f"Warmup   : {WARMUP} frames  |  Measure : {N_FRAMES} frames\n")

    frames = get_frames()
    print(f"Loaded {len(frames)} frames from {SEQUENCE}")

    print("\nLoading detector...")
    detector, det_tf = load_detector()
    gpu_cache = build_gpu_cache(frames, det_tf)
    results   = {}

    # Load estimators one at a time and delete after use to minimise peak
    # memory on Jetson (shared CPU/GPU RAM).

    (RESULTS_DIR / "timing.json").unlink(missing_ok=True)

    sampler = ResourceSampler(interval_ms=200)
    sampler.start_collecting()          # one tegrastats process for whole run

    markers          = {}               # condition → (t_start, t_end)
    resource_results = {}

    def _measure(cond_name, cond_label, lats_fn):
        if cond_name not in WANT:
            return
        print(f"\n{'='*60}")
        print(f"Condition: {cond_label}")
        print(f"{'='*60}")
        t0 = time.time();  write_marker(f"{cond_name}_start")
        lats = lats_fn()
        t1 = time.time();  write_marker(f"{cond_name}_end")

        results[cond_name]          = _stats(lats)
        resource_results[cond_name] = sampler.get_stats(t0, t1)
        markers[cond_name]          = (t0, t1)

        s = results[cond_name]
        r = resource_results[cond_name]
        print(f"  → {s['mean_ms']:.1f} ± {s['std_ms']:.1f} ms  "
              f"(p50={s['p50_ms']:.1f}  p95={s['p95_ms']:.1f})")
        print(f"     GPU={r.get('gpu_pct_mean','?')}%  "
              f"CPU={r.get('cpu_pct_mean','?')}%  "
              f"RAM={r.get('ram_mb_mean','?')}MB  "
              f"Power={r.get('power_mw_mean','?')}mW")
        torch.cuda.empty_cache()
        time.sleep(2)

    _measure("detector_only", "Detector only (FasterRCNN, GPU)",
             lambda: run_detector_only(detector, gpu_cache))

    if "iounet_gpu" in WANT:
        iounet_head = _load_mlp_head(IoUPredictor, IOU_CKPT)
        _measure("iounet_gpu", "Detector + IoUNet  (both GPU)",
                 lambda: run_neural_gpu("IoUNet", detector, iounet_head, gpu_cache))
        del iounet_head

    if "gfl_gpu" in WANT:
        gfl_head = _load_mlp_head(GFLPredictor, GFL_CKPT)
        _measure("gfl_gpu", "Detector + GFL     (both GPU)",
                 lambda: run_neural_gpu("GFL", detector, gfl_head, gpu_cache))
        del gfl_head

    if "ptc_cpu" in WANT:
        ptc_est = PTCEstimator(conf_thr=PTC_CONF_THR)
        _measure("ptc_cpu", "Detector (GPU) + PTC-IoU (CPU)",
                 lambda: run_ptc_cpu(detector, ptc_est, gpu_cache))
        del ptc_est

    if "xgboost_cpu" in WANT:
        xgb_pred, extractor, buf = load_xgboost()
        _measure("xgboost_cpu", "Detector (GPU) + XGBoost (CPU)",
                 lambda: run_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache))
        del xgb_pred, extractor, buf

    # ── Summary — latency ─────────────────────────────────────────────────────
    det_mean = results["detector_only"]["mean_ms"]
    print(f"\n{'='*60}")
    print("Summary — E2E per-frame latency (ms, detector included)")
    print(f"{'='*60}")
    print(f"  {'Condition':<22s}  {'Mean':>7s}  {'Std':>5s}  "
          f"{'p50':>7s}  {'p95':>7s}  {'vs det-only':>11s}")
    print(f"  {'-'*65}")
    for name, s in results.items():
        delta = s["mean_ms"] - det_mean
        sign  = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"  {name:<22s}  {s['mean_ms']:>6.1f}ms  {s['std_ms']:>4.1f}  "
              f"{s['p50_ms']:>6.1f}ms  {s['p95_ms']:>6.1f}ms  {sign:>9s}ms")

    # ── Summary — resources ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary — Resource usage (mean over measured frames)")
    print(f"{'='*60}")
    print(f"  {'Condition':<22s}  {'GPU%':>6}  {'CPU%':>6}  {'RAM MB':>8}  {'Power mW':>10}")
    print(f"  {'-'*57}")
    for name, r in resource_results.items():
        print(f"  {name:<22s}  "
              f"{r.get('gpu_pct_mean', float('nan')):>6.1f}  "
              f"{r.get('cpu_pct_mean', float('nan')):>6.1f}  "
              f"{r.get('ram_mb_mean',  float('nan')):>8.0f}  "
              f"{r.get('power_mw_mean',float('nan')):>10.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    sampler.stop_collecting()

    lat_out = RESULTS_DIR / "latency.json"
    lat_out.write_text(json.dumps(results, indent=2))

    # NOTE: write the in-process sampler's numbers to a SEPARATE file, not
    # resources.json. Inside the Docker container there is no tegrastats, so this
    # sampler only has CPU/RAM (via /proc) and no GPU/power. The authoritative
    # resources.json is produced by the HOST-side run_resources.py (which does
    # have tegrastats). Writing here to resources.json would race with — and
    # clobber — the host's GPU data when both run for Fig 3.
    res_out = RESULTS_DIR / "resources_inproc.json"
    res_out.write_text(json.dumps(resource_results, indent=2))

    print(f"\nLatency  → {lat_out}")
    print(f"Resources (in-process, CPU/RAM only) → {res_out}")

    # Skip Python/CUDA cleanup to avoid heap-corruption crash on Jetson.
    # All results are already written to disk above.
    import os; os._exit(0)


if __name__ == "__main__":
    main()
