# ─────────────────────────────────────────────────
#  detector/yolo_detector.py
#
#  Research-grade multi-pass detection pipeline.
#
#  Detection Strategy:
#
#  Pass 1 — Full display frame at imgsz=1280
#            Catches medium/large vehicles.
#
#  Pass 2 — 4 overlapping quarter-tiles (display frame)
#            Catches objects split across tile edges.
#
#  Pass 3 — Per-lane zone crop from the ORIGINAL 4K frame
#            upscaled to 1280×1280 at conf=0.03.
#            4K pixels → cars that are 13px at 720p
#            become 80px at 4K → detectable by YOLO.
#            Optional Real-ESRGAN SR applied here.
#
#  Fusion — Weighted Box Fusion (WBF) replaces NMS.
#            All passes merge via confidence-weighted
#            averaging, producing stable, non-jittery
#            bounding boxes.
#
#  Optimisation — FP16 half-precision on MPS/CUDA.
# ─────────────────────────────────────────────────

from ultralytics import YOLO
import cv2
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_PATH, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD,
                    LANE_ZONES, USE_FP16)

# ── Detection constants ────────────────────────────
YOLO_IMGSZ         = 1280
TILE_SIZE          = 640
CROP_IMGSZ         = 1280
CROP_CONF_THRESHOLD = 0.03
WBF_IOU_THRESHOLD  = 0.55   # WBF fusion threshold (higher = more merging)
WBF_SKIP_BOX_THR   = 0.0001 # discard boxes below this confidence before WBF


class YOLODetector:
    """
    Four-pass YOLOv8 detector with Weighted Box Fusion.

    Call detect(display_frame, native_frame) to get detections.
    If native_frame is None, falls back to display_frame for all passes.
    Returned bboxes are always in display_frame coordinate space.
    """

    CLASS_NAMES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[YOLODetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.vehicle_classes  = VEHICLE_CLASSES
        self.conf_threshold   = CONFIDENCE_THRESHOLD
        self._sr_enabled      = False   # toggled via /api/toggle_sr
        self._sr_enhancer     = None    # lazy-loaded on first toggle-on
        self._last_latency_ms = 0.0     # detection latency for benchmarking

        import torch
        if torch.backends.mps.is_available():
            self._device = "mps"
            print("[YOLODetector] Using Apple MPS (Metal GPU) ✅")
        elif torch.cuda.is_available():
            self._device = "cuda"
            print("[YOLODetector] Using CUDA GPU ✅")
        else:
            self._device = "cpu"
            print("[YOLODetector] Using CPU")

        self._use_fp16 = USE_FP16 and self._device in ("mps", "cuda")
        if self._use_fp16:
            print("[YOLODetector] FP16 half-precision enabled ✅")

        # Warmup — eliminates the first-frame latency spike
        self._warmup()
        print("[YOLODetector] Model ready.")

    # ── Public API ────────────────────────────────────────────

    def detect(self, display_frame, native_frame=None,
               frame_w: int = None, frame_h: int = None):
        """
        Run multi-pass detection with WBF fusion.

        Parameters
        ----------
        display_frame : np.ndarray  BGR at display resolution (720p)
        native_frame  : np.ndarray  BGR at native resolution (4K), optional

        Returns list of dicts: {bbox, cls_id, label, conf, cx, cy}
        All bboxes are in display_frame coordinate space.
        """
        t_start = time.perf_counter()

        dh, dw = display_frame.shape[:2]
        src  = native_frame if native_frame is not None else display_frame
        sh, sw = src.shape[:2]
        sx_n2d = dw / sw
        sy_n2d = dh / sh

        all_dets = []

        # ── Pass 1: full display frame ────────────────────────
        all_dets.extend(
            self._infer_patch(display_frame, 0, 0, dw, dh,
                              imgsz=YOLO_IMGSZ,
                              conf=self.conf_threshold,
                              ox=0, oy=0, sx=1.0, sy=1.0)
        )

        # ── Pass 2: 4 overlapping quarter-tiles (display frame) ──
        half_w  = dw // 2
        half_h  = dh // 2
        ov_x    = int(half_w * 0.20)
        ov_y    = int(half_h * 0.20)

        tiles = [
            (0,              0,              half_w + ov_x,  half_h + ov_y),
            (half_w - ov_x,  0,              dw,             half_h + ov_y),
            (0,              half_h - ov_y,  half_w + ov_x,  dh),
            (half_w - ov_x,  half_h - ov_y, dw,             dh),
        ]
        for (tx1, ty1, tx2, ty2) in tiles:
            crop = display_frame[ty1:ty2, tx1:tx2]
            ch, cw = crop.shape[:2]
            if ch <= 0 or cw <= 0:
                continue
            infer = cv2.resize(crop, (TILE_SIZE, TILE_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            all_dets.extend(
                self._infer_patch(infer, 0, 0, TILE_SIZE, TILE_SIZE,
                                  imgsz=TILE_SIZE,
                                  conf=self.conf_threshold,
                                  ox=tx1, oy=ty1,
                                  sx=cw / TILE_SIZE, sy=ch / TILE_SIZE)
            )

        # ── Pass 3: per-lane zone crops from NATIVE frame ─────
        for zone_name, frac_pts in LANE_ZONES.items():
            xs = [p[0] for p in frac_pts]
            ys = [p[1] for p in frac_pts]

            if zone_name == "South":
                PAD_X  = int(sw * 0.12)
                PAD_YU = int(sh * 0.22)
                PAD_YD = int(sh * 0.02)
                zone_conf = 0.03
            elif zone_name == "North":
                PAD_X  = int(sw * 0.06)
                PAD_YU = int(sh * 0.05)
                PAD_YD = int(sh * 0.05)
                zone_conf = 0.03
            else:
                PAD_X  = int(sw * 0.06)
                PAD_YU = int(sh * 0.08)
                PAD_YD = int(sh * 0.08)
                zone_conf = CROP_CONF_THRESHOLD

            x1n = max(0, int(min(xs) * sw) - PAD_X)
            y1n = max(0, int(min(ys) * sh) - PAD_YU)
            x2n = min(sw, int(max(xs) * sw) + PAD_X)
            y2n = min(sh, int(max(ys) * sh) + PAD_YD)

            if x2n <= x1n or y2n <= y1n:
                continue

            crop = src[y1n:y2n, x1n:x2n]
            ch, cw = crop.shape[:2]
            if ch <= 0 or cw <= 0:
                continue

            # Optional Real-ESRGAN super resolution
            if self._sr_enabled and self._sr_enhancer is not None:
                try:
                    crop = self._sr_enhancer.enhance(crop)
                    ch, cw = crop.shape[:2]
                except Exception:
                    pass  # SR failed — continue with original crop

            infer = cv2.resize(crop, (CROP_IMGSZ, CROP_IMGSZ),
                               interpolation=cv2.INTER_CUBIC)

            crop_sx = cw / CROP_IMGSZ
            crop_sy = ch / CROP_IMGSZ

            raw_dets = self._infer_patch(
                infer, 0, 0, CROP_IMGSZ, CROP_IMGSZ,
                imgsz=CROP_IMGSZ,
                conf=zone_conf,
                ox=0, oy=0, sx=1.0, sy=1.0
            )

            # Map infer → display
            for d in raw_dets:
                ix1, iy1, ix2, iy2 = d["bbox"]
                nx1 = ix1 * crop_sx + x1n
                ny1 = iy1 * crop_sy + y1n
                nx2 = ix2 * crop_sx + x1n
                ny2 = iy2 * crop_sy + y1n
                bx1 = max(0, min(dw-1, int(nx1 * sx_n2d)))
                by1 = max(0, min(dh-1, int(ny1 * sy_n2d)))
                bx2 = max(0, min(dw,   int(nx2 * sx_n2d)))
                by2 = max(0, min(dh,   int(ny2 * sy_n2d)))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                d["bbox"] = (bx1, by1, bx2, by2)
                d["cx"]   = (bx1 + bx2) // 2
                d["cy"]   = (by1 + by2) // 2

            all_dets.extend(raw_dets)

        # ── Weighted Box Fusion ───────────────────────────────
        fused = self._wbf(all_dets, dw, dh)

        self._last_latency_ms = (time.perf_counter() - t_start) * 1000
        return fused

    def toggle_sr(self) -> bool:
        """Toggle Real-ESRGAN super resolution on/off. Returns new state."""
        if not self._sr_enabled:
            # Lazy-load SR enhancer
            if self._sr_enhancer is None:
                try:
                    from detector.super_resolution import SuperResolutionEnhancer
                    self._sr_enhancer = SuperResolutionEnhancer(device=self._device)
                    print("[YOLODetector] SR enhancer loaded ✅")
                except Exception as e:
                    print(f"[YOLODetector] SR load failed: {e}")
                    return False
            self._sr_enabled = True
        else:
            self._sr_enabled = False
        print(f"[YOLODetector] Super Resolution: {'ON' if self._sr_enabled else 'OFF'}")
        return self._sr_enabled

    @property
    def latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def sr_enabled(self) -> bool:
        return self._sr_enabled

    # ── Internal helpers ──────────────────────────────────────

    def _warmup(self):
        """Run a dummy inference to pre-compile MPS/CUDA kernels."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.model(dummy, verbose=False, imgsz=640,
                       device=self._device,
                       half=self._use_fp16)
        except Exception:
            pass

    def _infer_patch(self, frame,
                     x1: int, y1: int, x2: int, y2: int,
                     imgsz: int, conf: float,
                     ox: int = 0, oy: int = 0,
                     sx: float = 1.0, sy: float = 1.0) -> list:
        """Run YOLO on frame[y1:y2, x1:x2]. Map bboxes to output space."""
        fh, fw = frame.shape[:2]
        crop   = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return []

        is_full = (x1 == 0 and y1 == 0 and x2 == fw and y2 == fh)
        if is_full:
            infer    = crop
            local_sx = sx
            local_sy = sy
            local_ox = ox
            local_oy = oy
        else:
            infer    = cv2.resize(crop, (imgsz, imgsz),
                                  interpolation=cv2.INTER_LINEAR)
            local_sx = (cw / imgsz) * sx
            local_sy = (ch / imgsz) * sy
            local_ox = int(x1 * sx) + ox
            local_oy = int(y1 * sy) + oy

        results = self.model(
            infer,
            verbose=False,
            imgsz=imgsz,
            device=self._device,
            conf=conf,
            half=self._use_fp16,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_classes:
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            out_x1 = int(bx1 * local_sx) + local_ox
            out_y1 = int(by1 * local_sy) + local_oy
            out_x2 = int(bx2 * local_sx) + local_ox
            out_y2 = int(by2 * local_sy) + local_oy

            if out_x2 <= out_x1 or out_y2 <= out_y1:
                continue

            cx = (out_x1 + out_x2) // 2
            cy = (out_y1 + out_y2) // 2
            cf = float(box.conf[0])

            detections.append({
                "bbox":   (out_x1, out_y1, out_x2, out_y2),
                "cls_id": cls_id,
                "label":  self.CLASS_NAMES.get(cls_id, "vehicle"),
                "conf":   round(cf, 2),
                "cx":     cx,
                "cy":     cy,
            })

        return detections

    def _wbf(self, detections: list, fw: int, fh: int) -> list:
        """
        Weighted Box Fusion — fuses overlapping detections from multi-pass
        inference into averaged, confidence-weighted boxes.

        Normalises to [0,1], calls ensemble_boxes.weighted_boxes_fusion,
        then maps back to pixel space.
        """
        if not detections:
            return []

        try:
            from ensemble_boxes import weighted_boxes_fusion
        except ImportError:
            # Fallback to greedy NMS if library not available
            return self._nms_fallback(detections)

        # Group by class (WBF is class-agnostic by default but we do per-class)
        boxes_list, scores_list, labels_list = [], [], []

        norm_boxes = []
        scores     = []
        labels     = []

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            # Normalise to [0, 1]
            nx1 = max(0.0, x1 / fw)
            ny1 = max(0.0, y1 / fh)
            nx2 = min(1.0, x2 / fw)
            ny2 = min(1.0, y2 / fh)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            norm_boxes.append([nx1, ny1, nx2, ny2])
            scores.append(d["conf"])
            labels.append(float(d["cls_id"]))

        if not norm_boxes:
            return []

        boxes_list  = [norm_boxes]
        scores_list = [scores]
        labels_list = [labels]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=WBF_IOU_THRESHOLD,
            skip_box_thr=WBF_SKIP_BOX_THR,
            conf_type="avg",
        )

        result = []
        for (nx1, ny1, nx2, ny2), cf, cls_id in zip(
                fused_boxes, fused_scores, fused_labels):
            bx1 = max(0, min(fw-1, int(nx1 * fw)))
            by1 = max(0, min(fh-1, int(ny1 * fh)))
            bx2 = max(0, min(fw,   int(nx2 * fw)))
            by2 = max(0, min(fh,   int(ny2 * fh)))
            if bx2 <= bx1 or by2 <= by1:
                continue
            cls_id = int(cls_id)
            result.append({
                "bbox":   (bx1, by1, bx2, by2),
                "cls_id": cls_id,
                "label":  self.CLASS_NAMES.get(cls_id, "vehicle"),
                "conf":   round(float(cf), 2),
                "cx":     (bx1 + bx2) // 2,
                "cy":     (by1 + by2) // 2,
            })

        return result

    def _nms_fallback(self, detections: list) -> list:
        """Greedy IoU-NMS fallback when ensemble_boxes is unavailable."""
        dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            bx1, by1, bx2, by2 = best["bbox"]
            remaining = []
            for d in dets:
                dx1, dy1, dx2, dy2 = d["bbox"]
                ix1 = max(bx1, dx1); iy1 = max(by1, dy1)
                ix2 = min(bx2, dx2); iy2 = min(by2, dy2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                ab    = (bx2-bx1) * (by2-by1)
                ad    = (dx2-dx1) * (dy2-dy1)
                union = ab + ad - inter
                iou   = inter / union if union > 0 else 0.0
                if iou < 0.45:
                    remaining.append(d)
            dets = remaining
        return kept
