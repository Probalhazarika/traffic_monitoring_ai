# ─────────────────────────────────────────────────
#  detector/yolo_detector.py
#
#  Detection Strategy — four-pass pipeline:
#
#  Pass 1 — Full display frame at imgsz=1280
#            (fast global pass on the 720p frame)
#
#  Pass 2 — 4 overlapping quarter-tiles on display frame
#            (catches objects split across tile edges)
#
#  Pass 3 — Per-lane zone crop from the ORIGINAL 4K frame,
#            upscaled to 1280×1280 at conf=0.03.
#            Running on the native-resolution frame means cars
#            that are only 13×13px at 720p become 80×80px at 4K —
#            large enough for YOLO to recognise confidently.
#
#  All passes merged and deduplicated via IoU-NMS.
#  All returned bbox coordinates are in display-frame space (720p).
# ─────────────────────────────────────────────────

from ultralytics import YOLO
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD, LANE_ZONES

# Full-frame inference size (used on the display-resolution frame)
YOLO_IMGSZ = 1280

# Tile size for Pass 2
TILE_SIZE = 640

# Zone crop upscale size — the native-res crop is upscaled to this
CROP_IMGSZ = 1280

# Confidence for zone crops (lower = catch dim/small cars)
CROP_CONF_THRESHOLD = 0.03

# IoU threshold for NMS deduplication across passes
NMS_IOU_THRESHOLD = 0.45


class YOLODetector:
    """
    Four-pass YOLOv8 detector optimised for overhead 4K traffic cameras.

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
        self.vehicle_classes = VEHICLE_CLASSES
        self.conf_threshold  = CONFIDENCE_THRESHOLD

        import torch
        if torch.backends.mps.is_available():
            self._device = "mps"
            print("[YOLODetector] Using Apple MPS (Metal GPU) ✅")
        else:
            self._device = "cpu"
            print("[YOLODetector] Using CPU (no MPS available)")
        print("[YOLODetector] Model ready.")

    # ── Public API ────────────────────────────────────────────

    def detect(self, display_frame, native_frame=None,
               frame_w: int = None, frame_h: int = None):
        """
        Run four-pass detection.

        Parameters
        ----------
        display_frame : np.ndarray
            BGR frame at display resolution (e.g. 1280×720).
            Passes 1 & 2 run on this.
        native_frame : np.ndarray, optional
            BGR frame at native (4K) resolution.
            Pass 3 zone crops run on this if provided.
            Falls back to display_frame if None.

        Returns list of dicts: {bbox, cls_id, label, conf, cx, cy}
        All bboxes are in display_frame coordinate space.
        """
        dh, dw = display_frame.shape[:2]
        src = native_frame if native_frame is not None else display_frame
        sh, sw = src.shape[:2]

        # Scale factors from native → display space
        sx_n2d = dw / sw   # native-x → display-x
        sy_n2d = dh / sh   # native-y → display-y

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
                                  sx=cw/TILE_SIZE, sy=ch/TILE_SIZE)
            )

        # ── Pass 3: per-lane zone crops from NATIVE frame ─────
        # Using the original 4K pixels gives YOLO 3-9× more pixel area
        # per vehicle compared to the 720p display frame.
        for zone_name, frac_pts in LANE_ZONES.items():
            xs = [p[0] for p in frac_pts]
            ys = [p[1] for p in frac_pts]

            if zone_name == "South":
                # South zone sits at the bottom edge; cars queue above.
                # Use large upward padding to capture cars approaching the line.
                PAD_X  = int(sw * 0.12)
                PAD_YU = int(sh * 0.22)   # big upward pad
                PAD_YD = int(sh * 0.02)
                zone_conf = 0.03
            elif zone_name == "North":
                # North zone: cars at far end are tiny even at 4K.
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

            # Upscale the native crop to CROP_IMGSZ for inference
            infer = cv2.resize(crop, (CROP_IMGSZ, CROP_IMGSZ),
                               interpolation=cv2.INTER_LINEAR)

            # Scale factors: infer → native → display
            # infer → crop: cw/CROP_IMGSZ, ch/CROP_IMGSZ
            # crop origin (native): x1n, y1n
            # native → display: sx_n2d, sy_n2d
            crop_sx = cw / CROP_IMGSZ
            crop_sy = ch / CROP_IMGSZ

            raw_dets = self._infer_patch(
                infer, 0, 0, CROP_IMGSZ, CROP_IMGSZ,
                imgsz=CROP_IMGSZ,
                conf=zone_conf,
                ox=0, oy=0, sx=1.0, sy=1.0   # coords in infer space
            )

            # Map infer → display
            for d in raw_dets:
                ix1, iy1, ix2, iy2 = d["bbox"]
                # infer → native crop
                nx1 = ix1 * crop_sx + x1n
                ny1 = iy1 * crop_sy + y1n
                nx2 = ix2 * crop_sx + x1n
                ny2 = iy2 * crop_sy + y1n
                # native → display
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

        # ── Deduplicate ───────────────────────────────────────
        return self._nms(all_dets, NMS_IOU_THRESHOLD)

    # ── Internal helpers ──────────────────────────────────────

    def _infer_patch(self, frame,
                     x1: int, y1: int, x2: int, y2: int,
                     imgsz: int, conf: float,
                     ox: int = 0, oy: int = 0,
                     sx: float = 1.0, sy: float = 1.0) -> list:
        """
        Run YOLO on `frame[y1:y2, x1:x2]` (or the whole `frame` when the
        region covers it entirely).  Returned bboxes are mapped back to the
        caller's coordinate space via (ox, oy, sx, sy):

            display_x = bbox_x * sx + ox
            display_y = bbox_y * sy + oy
        """
        fh, fw = frame.shape[:2]
        crop   = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return []

        is_full = (x1 == 0 and y1 == 0 and x2 == fw and y2 == fh)
        if is_full:
            infer     = crop
            local_sx  = sx
            local_sy  = sy
            local_ox  = ox
            local_oy  = oy
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
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.vehicle_classes:
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()

            # Map to output coordinate space
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

    @staticmethod
    def _nms(detections: list, iou_threshold: float) -> list:
        """Greedy IoU-NMS — keeps highest-confidence box when boxes overlap."""
        if not detections:
            return []

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
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ab    = (bx2 - bx1) * (by2 - by1)
                ad    = (dx2 - dx1) * (dy2 - dy1)
                union = ab + ad - inter
                iou   = inter / union if union > 0 else 0.0
                if iou < iou_threshold:
                    remaining.append(d)
            dets = remaining

        return kept
