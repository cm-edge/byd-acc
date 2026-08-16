#!/usr/bin/env python3
"""
Estimator-only latency benchmark on Jetson Orin Nano.

Measures the incremental inference overhead of each estimator AFTER the detector
has already run (detector latency is NOT included in any reported number).

Three estimators (weights shipped in weights/):
  - weights/neural/gfl.pth      : RoI pool + GFL 3-layer MLP     (GPU)
  - weights/neural/iou_net.pth  : RoI pool + IoUNet 3-layer MLP  (GPU)
  - weights/xgboost/predictor.pkl : XGBoost on hand-crafted features (CPU)

Neural estimators share the FasterRCNN backbone via a forward hook so the
backbone never runs twice. XGBoost operates on a random feature vector of the
same shape used at runtime (window_size * n_features) — accuracy is irrelevant
here; we only care about inference time.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "estimator"))

from iounet_inference import IoUPredictor, _roi_features, ROI_DIM, CONF_THR, MAX_BOXES
from gfl_inference import GFLPredictor

# ── Config ────────────────────────────────────────────────────────────────────
MOT17_ROOT  = ROOT / "data" / "MOT17"
SEQUENCE    = "MOT17-02-FRCNN"
N_FRAMES    = 150
WARMUP      = 20
RESULTS_DIR = Path(__file__).parent / "results"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU_CACHE = 30

GFL_CKPT = ROOT / "weights/neural/gfl.pth"
IOU_CKPT = ROOT / "weights/neural/iou_net.pth"
XGB_CKPT = ROOT / "weights/xgboost/predictor.pkl"
XGB_N_FEATURES = 196  # window=7 x 28 features

_KEY_MAP = {"fc1": "net.0", "fc2": "net.2", "fc3": "net.4"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_detector():
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(DEVICE).eval()
    return model, weights.transforms()


def _load_mlp(cls, ckpt_path):
    head = cls(ROI_DIM).to(DEVICE).eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        mapped = {}
        for k, v in ckpt.items():
            prefix = k.split(".")[0]
            mapped[_KEY_MAP.get(prefix, prefix) + "." + ".".join(k.split(".")[1:])] = v
        head.load_state_dict(mapped)
        print(f"  Loaded weights: {ckpt_path.name}")
    else:
        print(f"  WARNING: {ckpt_path.name} not found — random init (timing only)")
    return head


def load_gfl_head():
    return _load_mlp(GFLPredictor, GFL_CKPT)


def load_iou_head():
    return _load_mlp(IoUPredictor, IOU_CKPT)


def load_xgboost():
    with open(XGB_CKPT, "rb") as f:
        bundle = pickle.load(f)
    booster = bundle["model"].get_booster()
    print(f"  Loaded XGBoost booster from {XGB_CKPT.name}  (n_features={XGB_N_FEATURES})")
    return booster


# ── Frame helpers ─────────────────────────────────────────────────────────────

def get_frames():
    img_dir = MOT17_ROOT / "train" / SEQUENCE / "img1"
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        sys.exit(f"No frames found in {img_dir}")
    return frames


def build_gpu_cache(frames, det_tf):
    n = min(N_GPU_CACHE, len(frames))
    print(f"  Preloading {n} frames to GPU...", end=" ", flush=True)
    cache = [det_tf(Image.open(p).convert("RGB")).to(DEVICE) for p in frames[:n]]
    torch.cuda.synchronize()
    print("done.")
    return cache


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _run_neural_estimator(name, detector, head, gpu_cache, n, warmup):
    """
    Time: RoI pool + MLP head only.
    The detector runs every iteration to produce boxes + backbone features,
    but its time is NOT recorded — only the estimator's.
    """
    latencies = []
    total = warmup + n
    _feat_cache: dict = {}
    hook = detector.backbone.register_forward_hook(
        lambda m, inp, out: _feat_cache.__setitem__("features", out)
    )
    try:
        for i in range(total):
            det_t = gpu_cache[i % len(gpu_cache)]
            h, w = det_t.shape[-2], det_t.shape[-1]

            # Detector forward — backbone features captured by hook (NOT timed)
            with torch.no_grad():
                det_out = detector([det_t])
            torch.cuda.synchronize()

            # Estimator: RoI pool + MLP head (TIMED)
            t0 = time.perf_counter()
            with torch.no_grad():
                boxes  = det_out[0]["boxes"]
                scores = det_out[0]["scores"]
                if len(boxes) > 0:
                    keep  = scores >= CONF_THR
                    boxes = boxes[keep][:MAX_BOXES]
                if len(boxes) > 0:
                    roi = _roi_features(
                        detector, det_t.unsqueeze(0),
                        [boxes], [(h, w)],
                        cached_features=_feat_cache.get("features"),
                    )
                    _ = head(roi)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000

            if i >= warmup:
                latencies.append(ms)
            if (i + 1) % 50 == 0:
                print(f"  [{name}] {i+1}/{total}  est={ms:.2f}ms")
    finally:
        hook.remove()
    return latencies


def _run_xgboost_estimator(booster, n, warmup):
    """
    Time: XGBoost Booster.predict on a random DMatrix.
    Shape matches runtime input: (1, n_features).
    """
    rng = np.random.default_rng(42)
    latencies = []
    total = warmup + n

    for i in range(total):
        X = rng.random((1, XGB_N_FEATURES)).astype(np.float32)
        t0 = time.perf_counter()
        _ = booster.inplace_predict(X)
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            latencies.append(ms)
        if (i + 1) % 50 == 0:
            print(f"  [XGBoost] {i+1}/{total}  est={ms:.3f}ms")

    return latencies


def _stats(lats):
    return {
        "mean_ms":  float(np.mean(lats)),
        "std_ms":   float(np.std(lats)),
        "p50_ms":   float(np.percentile(lats, 50)),
        "p95_ms":   float(np.percentile(lats, 95)),
        "n":        len(lats),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = get_frames()
    print(f"Sequence : {SEQUENCE}  ({len(frames)} frames)")
    print(f"Device   : {DEVICE}\n")

    print("Loading detector (FasterRCNN-MobileNetV3-FPN)...")
    detector, det_tf = load_detector()

    print("Loading GFL head...")
    gfl_head = load_gfl_head()

    print("Loading IoUNet head...")
    iou_head = load_iou_head()

    print("Loading XGBoost booster...")
    xgb_clf = load_xgboost()

    gpu_cache = build_gpu_cache(frames, det_tf)

    results = {}

    # ── GFL ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GFL estimator (RoI pool + 3-layer MLP, GPU)")
    print(f"{'='*60}")
    lats = _run_neural_estimator("GFL", detector, gfl_head, gpu_cache, N_FRAMES, WARMUP)
    results["gfl_qfl_quality_head"] = _stats(lats)
    s = results["gfl_qfl_quality_head"]
    print(f"  → {s['mean_ms']:.2f} ± {s['std_ms']:.2f} ms  (p95={s['p95_ms']:.2f}ms)")

    # ── IoUNet ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("IoUNet estimator (RoI pool + 3-layer MLP, GPU)")
    print(f"{'='*60}")
    lats = _run_neural_estimator("IoUNet", detector, iou_head, gpu_cache, N_FRAMES, WARMUP)
    results["iou_net"] = _stats(lats)
    s = results["iou_net"]
    print(f"  → {s['mean_ms']:.2f} ± {s['std_ms']:.2f} ms  (p95={s['p95_ms']:.2f}ms)")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"XGBoost estimator (n_features={XGB_N_FEATURES}, CPU)")
    print(f"{'='*60}")
    lats = _run_xgboost_estimator(xgb_clf, N_FRAMES, WARMUP)
    results["xgboost"] = _stats(lats)
    s = results["xgboost"]
    print(f"  → {s['mean_ms']:.3f} ± {s['std_ms']:.3f} ms  (p95={s['p95_ms']:.3f}ms)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary — estimator overhead (detector NOT included)")
    print(f"{'='*60}")
    for name, s in results.items():
        print(f"  {name:<30s}  {s['mean_ms']:7.3f} ms  ± {s['std_ms']:.3f}  p95={s['p95_ms']:.3f}ms")

    out = RESULTS_DIR / "estimator_latency.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
