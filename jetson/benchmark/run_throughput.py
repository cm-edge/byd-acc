#!/usr/bin/env python3
"""
Throughput benchmark — Jetson Orin Nano.

Measures maximum sustained frames-per-second for each condition.
200 frames, sent as fast as possible (overloaded / no throttling).

GPU baselines (iounet_gpu, gfl_gpu):
  Sequential pipeline — detector then estimator back-to-back on GPU.
  GPU serialises both, so throughput = 1 / (det_time + est_time).

CPU estimators (ptc_cpu, xgboost_cpu):
  Two sub-conditions measured:
    _seq       — sequential: detector then CPU estimator, no overlap
    _pipelined — threaded producer-consumer: GPU runs det[N] while
                 CPU runs est[N-1] concurrently. GPU is the bottleneck;
                 CPU estimator adds near-zero latency to throughput.

Results saved to jetson/benchmark/results/throughput.json.
"""

import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

# sklearn (in /host-packages) needs real scipy — don't mock scipy at all.
_HOST_PKGS = "/host-packages"
if _HOST_PKGS not in sys.path:
    sys.path.append(_HOST_PKGS)

import os

_ROOT_EARLY = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME",
                      str(_ROOT_EARLY / "data" / "torch_hub"))

# Import the XGBoost stack BEFORE torch: importing pandas/sklearn after torch
# has been running a frame loop corrupts the glibc heap on Jetson ("corrupted
# size vs. prev_size" abort in the xgboost conditions). Harmless no-op if
# the packages are not installed.
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
# MOT17 benchmark → use the dataset-correct box threshold 0.1 (CONF_THR_MOT17),
# not the generic 0.5. See run_latency.py for the full rationale.
CONF_THR = CONF_THR_MOT17
from gfl_inference import GFLPredictor
from ptc_inference import PTCEstimator
from feature_extractor import FrameFeatureExtractor, build_window_vector
# DifficultyPredictor imported lazily inside load_xgboost() to avoid
# pulling sklearn/scipy at module level.

# ── Config ────────────────────────────────────────────────────────────────────
MOT17_ROOT  = ROOT / "data" / "MOT17"
SEQUENCE    = "MOT17-02-FRCNN"
N_FRAMES    = 200
WARMUP      = 20
RESULTS_DIR = Path(__file__).parent / "results"


def write_marker(key: str):
    f = RESULTS_DIR / "timing_throughput.json"
    markers = json.loads(f.read_text()) if f.exists() else {}
    markers[key] = time.time()
    f.write_text(json.dumps(markers, indent=2))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU_CACHE = 30

IOU_CKPT = ROOT / "weights/neural/iou_net.pth"
GFL_CKPT = ROOT / "weights/neural/gfl.pth"
XGB_CKPT = ROOT / "weights/xgboost/predictor.pkl"

XGB_WINDOW   = 7
IMG_W, IMG_H = 1920, 1080
VEHICLE_IDS  = {3, 6, 8}
PTC_CONF_THR = 0.5

_KEY_MAP = {"fc1": "net.0", "fc2": "net.2", "fc3": "net.4"}

_SENTINEL = object()   # signals consumer thread to stop


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_detector():
    # Full-resolution FasterRCNN-MobileNetV3-L-FPN — same model the rest of the
    # repo uses. The detector variant does not affect the paper's relative
    # estimator-overhead conclusions. DETECTOR_RES=320 exists only for exploring
    # the detector floor, not for reproduction.
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


def _det_to_ptc(det_out):
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


def _fps_stats(elapsed, n):
    return {"fps": float(n / elapsed), "elapsed_s": float(elapsed), "n_frames": n}


# ── Sequential runners ────────────────────────────────────────────────────────

def _warmup(detector, gpu_cache):
    print(f"  Warming up ({WARMUP} frames)...", end=" ", flush=True)
    for i in range(WARMUP):
        with torch.no_grad():
            _ = detector([gpu_cache[i % len(gpu_cache)]])
    torch.cuda.synchronize()
    print("done.")


def run_seq_neural_gpu(name, detector, head, gpu_cache):
    """Sequential: Detector + GPU head. Backbone shared via hook — runs once per frame."""
    _warmup(detector, gpu_cache)
    _feat_cache = {}
    hook = detector.backbone.register_forward_hook(
        lambda m, inp, out: _feat_cache.__setitem__("features", out)
    )
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    try:
        for i in range(N_FRAMES):
            det_t = gpu_cache[i % len(gpu_cache)]
            h, w  = det_t.shape[-2], det_t.shape[-1]
            with torch.no_grad():
                det_out = detector([det_t])
                boxes   = det_out[0]["boxes"]
                scores  = det_out[0]["scores"]
                keep    = scores >= CONF_THR
                boxes   = boxes[keep][:MAX_BOXES]
                if len(boxes) > 0:
                    roi = _roi_features(
                        detector, det_t.unsqueeze(0),
                        [boxes], [(h, w)],
                        cached_features=_feat_cache.get("features"),
                    )
                    _ = head(roi)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{N_FRAMES}")
    finally:
        hook.remove()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    fps = N_FRAMES / elapsed
    print(f"  → {fps:.2f} fps  ({elapsed:.1f}s)")
    return fps, elapsed


def run_seq_ptc_cpu(detector, ptc_est, gpu_cache):
    """Sequential: Detector (GPU) then PTC (CPU), no overlap."""
    ptc_est.reset()
    _warmup(detector, gpu_cache)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(N_FRAMES):
        det_t = gpu_cache[i % len(gpu_cache)]
        with torch.no_grad():
            det_out = detector([det_t])
        torch.cuda.synchronize()
        dets = _det_to_ptc(det_out[0])
        _ = ptc_est.update(dets)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N_FRAMES}")
    elapsed = time.perf_counter() - t_start
    fps = N_FRAMES / elapsed
    print(f"  → {fps:.2f} fps  ({elapsed:.1f}s)")
    return fps, elapsed


def run_seq_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache):
    """Sequential: Detector (GPU) then XGBoost (CPU), no overlap."""
    extractor.reset()
    buf.clear()
    _warmup(detector, gpu_cache)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for i in range(N_FRAMES):
        det_t = gpu_cache[i % len(gpu_cache)]
        with torch.no_grad():
            det_out = detector([det_t])
        torch.cuda.synchronize()
        boxes  = det_out[0]["boxes"].cpu().numpy()
        scores = det_out[0]["scores"].cpu().numpy()
        labels = det_out[0]["labels"].cpu().numpy()
        feat   = extractor.update(boxes, scores, labels, VEHICLE_IDS)
        buf.append(feat)
        if len(buf) == XGB_WINDOW:
            X = build_window_vector(list(buf), xgb_pred.feature_columns)
            _ = xgb_pred.predict(X)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N_FRAMES}")
    elapsed = time.perf_counter() - t_start
    fps = N_FRAMES / elapsed
    print(f"  → {fps:.2f} fps  ({elapsed:.1f}s)")
    return fps, elapsed


# ── Pipelined runners (CPU estimator overlaps GPU detector) ───────────────────
#
# Thread layout:
#   GPU thread (producer) : detector([frame]) → puts det_out on queue
#   CPU thread (consumer) : pulls det_out, runs CPU estimator
#
# The queue depth is 1 so the producer never races far ahead.
# Wall-clock throughput = N_FRAMES / (total elapsed from first frame start
#                                     to last frame finished).

def run_pipelined_ptc_cpu(detector, ptc_est, gpu_cache):
    """Pipelined: GPU detector + CPU PTC running concurrently via threads."""
    ptc_est.reset()
    _warmup(detector, gpu_cache)

    det_queue: queue.Queue = queue.Queue(maxsize=2)

    def gpu_producer():
        for i in range(N_FRAMES):
            det_t = gpu_cache[i % len(gpu_cache)]
            with torch.no_grad():
                det_out = detector([det_t])
            torch.cuda.synchronize()
            det_queue.put(det_out[0])
            if (i + 1) % 50 == 0:
                print(f"    [GPU] {i+1}/{N_FRAMES}")
        det_queue.put(_SENTINEL)

    def cpu_consumer():
        while True:
            item = det_queue.get()
            if item is _SENTINEL:
                break
            dets = _det_to_ptc(item)
            _ = ptc_est.update(dets)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    producer = threading.Thread(target=gpu_producer, daemon=True)
    consumer = threading.Thread(target=cpu_consumer, daemon=True)
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()

    elapsed = time.perf_counter() - t_start
    fps = N_FRAMES / elapsed
    print(f"  → {fps:.2f} fps  ({elapsed:.1f}s)")
    return fps, elapsed


def run_pipelined_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache):
    """Pipelined: GPU detector + CPU XGBoost running concurrently via threads."""
    extractor.reset()
    buf.clear()
    _warmup(detector, gpu_cache)

    det_queue: queue.Queue = queue.Queue(maxsize=2)

    def gpu_producer():
        for i in range(N_FRAMES):
            det_t = gpu_cache[i % len(gpu_cache)]
            with torch.no_grad():
                det_out = detector([det_t])
            torch.cuda.synchronize()
            # Transfer results to CPU before enqueuing so GPU is free immediately
            det_queue.put({
                "boxes":  det_out[0]["boxes"].cpu().numpy(),
                "scores": det_out[0]["scores"].cpu().numpy(),
                "labels": det_out[0]["labels"].cpu().numpy(),
            })
            if (i + 1) % 50 == 0:
                print(f"    [GPU] {i+1}/{N_FRAMES}")
        det_queue.put(_SENTINEL)

    def cpu_consumer():
        while True:
            item = det_queue.get()
            if item is _SENTINEL:
                break
            feat = extractor.update(item["boxes"], item["scores"],
                                    item["labels"], VEHICLE_IDS)
            buf.append(feat)
            if len(buf) == XGB_WINDOW:
                X = build_window_vector(list(buf), xgb_pred.feature_columns)
                _ = xgb_pred.predict(X)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    producer = threading.Thread(target=gpu_producer, daemon=True)
    consumer = threading.Thread(target=cpu_consumer, daemon=True)
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()

    elapsed = time.perf_counter() - t_start
    fps = N_FRAMES / elapsed
    print(f"  → {fps:.2f} fps  ({elapsed:.1f}s)")
    return fps, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import argparse
    _ALL = ["iounet_gpu", "gfl_gpu", "ptc_cpu_seq", "ptc_cpu_pipelined",
            "xgboost_cpu_seq", "xgboost_cpu_pipelined"]
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--conditions", nargs="+", default=_ALL,
                     help="subset of conditions to run (default: all). "
                          "XGBoost load can segfault after other conditions on Jetson; "
                          "run it alone with '--conditions xgboost_cpu_seq xgboost_cpu_pipelined'.")
    _ap.add_argument("--merge", action="store_true",
                     help="merge into existing throughput.json instead of overwriting")
    WANT = set(_ap.parse_args().conditions)

    print(f"Sequence : {SEQUENCE}")
    print(f"Device   : {DEVICE}")
    print(f"Frames   : {N_FRAMES}  (warmup={WARMUP})\n")

    frames = get_frames()

    # Load detector once — shared across all conditions.
    print("Loading detector...")
    detector, det_tf = load_detector()
    gpu_cache = build_gpu_cache(frames, det_tf)
    results   = {}

    # Each condition loads its estimator on demand and deletes it afterwards
    # to keep peak GPU/CPU memory as low as possible on Jetson (shared CPU/GPU RAM).

    (RESULTS_DIR / "timing_throughput.json").unlink(missing_ok=True)

    def _run(key, label, runner_fn):
        print(f"\n{'='*60}")
        print(f"Condition: {label}")
        print(f"{'='*60}")
        write_marker(f"{key}_start")
        fps, elapsed = runner_fn()   # runners now return (fps, elapsed_s)
        write_marker(f"{key}_end")
        results[key] = {
            "fps":           round(fps, 4),
            "elapsed_s":     round(elapsed, 3),
            "frame_ms_mean": round(1000.0 / fps, 3),
            "n_frames":      N_FRAMES,
        }
        torch.cuda.empty_cache()
        time.sleep(2)

    # ── IoUNet (GPU) ──────────────────────────────────────────────────────────
    if "iounet_gpu" in WANT:
        iounet_head = _load_mlp_head(IoUPredictor, IOU_CKPT)
        _run("iounet_gpu", "Detector + IoUNet  (both GPU, sequential)",
             lambda: run_seq_neural_gpu("IoUNet", detector, iounet_head, gpu_cache))
        del iounet_head

    # ── GFL (GPU) ─────────────────────────────────────────────────────────────
    if "gfl_gpu" in WANT:
        gfl_head = _load_mlp_head(GFLPredictor, GFL_CKPT)
        _run("gfl_gpu", "Detector + GFL     (both GPU, sequential)",
             lambda: run_seq_neural_gpu("GFL", detector, gfl_head, gpu_cache))
        del gfl_head

    # ── PTC (CPU) — sequential + pipelined ───────────────────────────────────
    if {"ptc_cpu_seq", "ptc_cpu_pipelined"} & WANT:
        ptc_est = PTCEstimator(conf_thr=PTC_CONF_THR)
        if "ptc_cpu_seq" in WANT:
            _run("ptc_cpu_seq",      "Detector (GPU) + PTC-IoU (CPU, sequential)",
                 lambda: run_seq_ptc_cpu(detector, ptc_est, gpu_cache))
        if "ptc_cpu_pipelined" in WANT:
            _run("ptc_cpu_pipelined","Detector (GPU) + PTC-IoU (CPU, pipelined)",
                 lambda: run_pipelined_ptc_cpu(detector, ptc_est, gpu_cache))
        del ptc_est

    # ── XGBoost (CPU) — sequential + pipelined ───────────────────────────────
    if {"xgboost_cpu_seq", "xgboost_cpu_pipelined"} & WANT:
        xgb_pred, extractor, buf = load_xgboost()
        if "xgboost_cpu_seq" in WANT:
            _run("xgboost_cpu_seq",      "Detector (GPU) + XGBoost (CPU, sequential)",
                 lambda: run_seq_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache))
        if "xgboost_cpu_pipelined" in WANT:
            _run("xgboost_cpu_pipelined","Detector (GPU) + XGBoost (CPU, pipelined)",
                 lambda: run_pipelined_xgboost_cpu(detector, xgb_pred, extractor, buf, gpu_cache))
        del xgb_pred, extractor, buf

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary — Throughput (fps = req/sec)")
    print(f"{'='*60}")
    gpu_ref = results.get("iounet_gpu", {}).get("fps", 1.0)
    print(f"  {'Condition':<28s}  {'fps':>6s}  {'ms/frame':>9s}  {'speedup':>8s}")
    print(f"  {'-'*56}")
    for key, vals in results.items():
        speedup = vals["fps"] / gpu_ref
        print(f"  {key:<28s}  {vals['fps']:>5.2f}  "
              f"{vals['frame_ms_mean']:>7.1f}ms  {speedup:>7.2f}×")

    out = RESULTS_DIR / "throughput.json"
    if _ap.parse_args().merge and out.exists():
        merged = json.loads(out.read_text())
        merged.update(results)
        results = merged
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out}")
    import os; os._exit(0)


if __name__ == "__main__":
    main()
