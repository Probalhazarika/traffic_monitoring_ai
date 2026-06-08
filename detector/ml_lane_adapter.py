# ─────────────────────────────────────────────────────────────────────────────
#  detector/ml_lane_adapter.py
#  Adapter that wraps the ML-based SegFormer/Mask2Former lane detection
#  pipeline into the same interface as the existing LaneDetector.
#
#  This is a drop-in replacement that the VideoProcessor uses when
#  ML_LANE_ENABLED = True in config.py.
#
#  Exposes the same public API as LaneDetector:
#    • calibrate(frames, video_path)     → runs model on first frame
#    • draw_lanes(frame) → np.ndarray
#    • assign_vehicles_to_lanes(dets, frame_w, frame_h) → dict
#    • get_polygons_serializable() → dict
#    • _refresh_polys(fw, fh)            → no-op (compatibility shim)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy imports (only pull in heavy deps when ML is enabled)
# ─────────────────────────────────────────────────────────────────────────────

def _lazy_import_ml():
    from utils.augmentations import get_val_transforms
    from models import build_model
    from lane_graph import LaneGraph
    from lane_tracker import LaneTracker
    from yolo_lane_fusion import YOLOLaneFusion
    return get_val_transforms, build_model, LaneGraph, LaneTracker, YOLOLaneFusion


# ─────────────────────────────────────────────────────────────────────────────
#  ML Lane Adapter
# ─────────────────────────────────────────────────────────────────────────────

class MLLaneAdapter:
    """
    Drop-in replacement for LaneDetector, backed by a trained SegFormer-B5
    or native Mask2Former model.

    Matches the exact public API of detector/lane_detector.py:LaneDetector.
    """

    LANE_COLORS = [
        (0,   255,   0),   # green
        (255, 165,   0),   # orange
        (0,   165, 255),   # blue-orange
        (255,   0, 255),   # magenta
        (0,   255, 255),   # cyan
        (255, 255,   0),   # yellow
        (255,  80,  80),   # salmon
        (80,  255,  80),   # lime
    ]

    def __init__(self):
        from config import (
            ML_LANE_MODEL_TYPE, ML_LANE_CHECKPOINT, ML_LANE_CONFIG,
            ML_LANE_IMAGE_SIZE, ML_LANE_THRESHOLD, ML_LANE_TRACKER,
            ML_LANE_MIN_LENGTH, ML_LANE_MAX_DIST,
        )

        self.model_type  = ML_LANE_MODEL_TYPE
        self.checkpoint  = ML_LANE_CHECKPOINT
        self.config_path = ML_LANE_CONFIG
        self.image_size  = ML_LANE_IMAGE_SIZE
        self.threshold   = ML_LANE_THRESHOLD
        self.use_tracker = ML_LANE_TRACKER
        self.min_length  = ML_LANE_MIN_LENGTH
        self.max_dist    = ML_LANE_MAX_DIST

        # State
        self._model       = None
        self._transform   = None
        self._device      = None
        self._lane_graph  = None
        self._tracker     = None
        self._confirmed_lanes = {}     # lane_id → LaneData
        self._lock        = threading.Lock()
        self._calibrated  = False

        print("[MLLaneAdapter] Initialised. Model will load on first calibrate().")

    # ── Device detection ──────────────────────────────────────────────────

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazily load the model on first calibrate() call."""
        get_val_transforms, build_model, LaneGraph, LaneTracker, _ = _lazy_import_ml()

        import yaml
        cfg: dict = {
            "model_type":  self.model_type,
            "backbone":    "nvidia/mit-b5",
            "num_classes": 2,
            "image_size":  self.image_size,
        }
        if Path(self.config_path).exists():
            with open(self.config_path) as f:
                cfg.update(yaml.safe_load(f))

        self._device    = self._get_device()
        print(f"[MLLaneAdapter] Loading {self.model_type} on {self._device}…")

        self._model = build_model(self.model_type, cfg).to(self._device)
        self._model.eval()

        ckpt_path = Path(self.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            self._model.load_state_dict(ckpt["model_state"])
            ep = ckpt.get("epoch", "?")
            iou = ckpt.get("best_iou", "?")
            print(f"[MLLaneAdapter] Checkpoint loaded (epoch={ep}, best_iou={iou})")
        else:
            print(f"[MLLaneAdapter] WARNING: No checkpoint at {ckpt_path}.")
            print("  Run: python3 train.py   to train the model first.")
            print("  Using random weights — detections will be meaningless.")

        self._transform = get_val_transforms(self.image_size)
        self._lane_graph = LaneGraph(
            min_lane_length=self.min_length,
            spline_points=200,
        )
        if self.use_tracker:
            self._tracker = LaneTracker(
                ema_alpha=0.3,
                max_age=8,
                min_hits=2,
                max_centroid_dist=200.0,
            )

        print(f"[MLLaneAdapter] Ready.")

    # ── Inference ─────────────────────────────────────────────────────────

    def _infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run ML inference on a BGR frame.

        Returns
        -------
        binary_mask : (H, W) uint8 {0, 255}  at original frame resolution
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        frame_rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        aug    = self._transform(image=frame_rgb)
        tensor = aug["image"].unsqueeze(0).to(self._device)

        with torch.no_grad():
            if self.model_type == "segformer":
                logits = self._model(tensor)
                probs  = torch.sigmoid(logits).squeeze()
            else:
                output = self._model(tensor)
                masks  = output["pred_masks"]
                h, w   = tensor.shape[-2:]
                masks  = F.interpolate(masks, size=(h, w),
                                       mode="bilinear", align_corners=False)
                probs, _ = torch.sigmoid(masks).squeeze(0).max(dim=0)

        prob_np = probs.cpu().float().numpy()

        # Upsample to original resolution
        if prob_np.shape != (orig_h, orig_w):
            prob_np = cv2.resize(prob_np, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)

        return (prob_np >= self.threshold).astype(np.uint8) * 255

    # ─────────────────────────────────────────────────────────────────────
    #  Public API — matches LaneDetector interface exactly
    # ─────────────────────────────────────────────────────────────────────

    def calibrate(self, frames: list, video_path: str = "") -> None:
        """
        Initialise the ML model and run the first inference to detect lanes.
        Called by VideoProcessor at startup with the first CALIBRATION_FRAMES.
        """
        if not frames:
            print("[MLLaneAdapter] calibrate(): no frames received.")
            return

        # Load model (once)
        if self._model is None:
            self._load_model()

        # Run inference on median-blended calibration frame
        print(f"[MLLaneAdapter] Running initial lane detection on "
              f"{len(frames)} calibration frames…")
        t0 = time.time()

        # Blend N frames for a cleaner road surface
        n     = min(len(frames), 30)
        stack = np.stack([frames[i] for i in range(n)], axis=0).astype(np.float32)
        blend = np.median(stack, axis=0).astype(np.uint8)

        binary = self._infer(blend)
        from lane_graph import LaneGraph
        raw_lanes = self._lane_graph.extract(binary)

        if self._tracker:
            # Seed the tracker with 5 identical frames so tracks confirm quickly
            for _ in range(5):
                confirmed = self._tracker.update(raw_lanes)
        else:
            confirmed = {lid: ld for lid, ld in raw_lanes.items()}

        with self._lock:
            self._confirmed_lanes = confirmed
            self._calibrated = True

        elapsed = time.time() - t0
        print(f"[MLLaneAdapter] Detected {len(confirmed)} lanes in {elapsed:.1f}s")
        for lid, lane in confirmed.items():
            print(f"  Lane {lid}: dir={lane.direction:.1f}°  "
                  f"len={lane.length:.0f}px")

    def update(self, frame_bgr: np.ndarray) -> None:
        """
        Process a new frame — updates confirmed lanes via tracker.
        Call this from the detector worker thread every frame.
        """
        if self._model is None or not self._calibrated:
            return

        try:
            binary    = self._infer(frame_bgr)
            raw_lanes = self._lane_graph.extract(binary)

            if self._tracker:
                confirmed = self._tracker.update(raw_lanes)
            else:
                confirmed = {lid: ld for lid, ld in raw_lanes.items()}

            with self._lock:
                if confirmed:
                    self._confirmed_lanes = confirmed

        except Exception as e:
            print(f"[MLLaneAdapter] update() error: {e}")

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw lane overlays on the frame.
        Matches the LaneDetector.draw_lanes() signature exactly.
        """
        with self._lock:
            lanes = dict(self._confirmed_lanes)

        if not lanes:
            return frame

        out = frame.copy()
        for lid, lane in lanes.items():
            color = self.LANE_COLORS[(lid - 1) % len(self.LANE_COLORS)]

            # Draw spline centerline
            if len(lane.spline_x) > 1:
                sx = np.array([int(x) for x in lane.spline_x])
                sy = np.array([int(y) for y in lane.spline_y])

                # Semi-transparent band
                band_up   = np.stack([sx, np.maximum(0, sy - 5)], 1)
                band_down = np.stack([sx, np.minimum(frame.shape[0]-1, sy + 5)], 1)
                hull      = np.vstack([band_up, band_down[::-1]]).reshape(-1, 1, 2).astype(np.int32)
                overlay   = out.copy()
                cv2.fillPoly(overlay, [hull], color)
                out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)

                # Centerline
                pts = np.stack([sx, sy], axis=1).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)

            # Lane ID label
            if len(lane.spline_x) > 0:
                mid = len(lane.spline_x) // 2
                cx  = int(lane.spline_x[mid])
                cy  = int(lane.spline_y[mid])
                cv2.putText(out, f"L{lid}", (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return out

    def assign_vehicles_to_lanes(self, detections: list,
                                  frame_w: int = 0, frame_h: int = 0) -> dict:
        """
        Assign detected vehicles to lanes by nearest-spline distance.
        Returns dict matching LaneDetector.assign_vehicles_to_lanes() format:
            {lane_name: [det, det, ...]}
        """
        from lane_graph import LaneGraph

        with self._lock:
            lanes = dict(self._confirmed_lanes)

        result: dict = {f"Lane {lid}": [] for lid in lanes}

        for det in detections:
            bbox = det.get("bbox", (0, 0, 0, 0))
            bx   = (bbox[0] + bbox[2]) // 2
            by   = bbox[3]   # bottom-centre

            lid, dist = LaneGraph.nearest_lane((bx, by), lanes,
                                               max_dist=self.max_dist)
            if lid is not None:
                result.setdefault(f"Lane {lid}", []).append(det)

        return result

    def get_polygons_serializable(self) -> dict:
        """
        Return lane representation for the shared state dict.
        Encodes each lane's spline as a polyline of [x, y] pairs.
        """
        with self._lock:
            lanes = dict(self._confirmed_lanes)

        polygons = {}
        for lid, lane in lanes.items():
            key = f"Lane {lid}"
            if lane.spline_x:
                polygons[key] = [
                    [int(x), int(y)]
                    for x, y in zip(lane.spline_x, lane.spline_y)
                ]
            else:
                polygons[key] = [[int(p[0]), int(p[1])] for p in lane.polyline]

        return polygons

    def _refresh_polys(self, fw: int, fh: int) -> None:
        """Compatibility shim — no-op for ML adapter."""
        pass

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def num_lanes(self) -> int:
        with self._lock:
            return len(self._confirmed_lanes)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
