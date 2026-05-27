# ─────────────────────────────────────────────────
#  detector/vehicle_counter.py
#
#  Counts vehicles per lane, computes hybrid density,
#  and annotates the frame with bounding boxes.
#
#  Hybrid Density Score
#  ────────────────────
#  Combines three signals for a more robust estimate:
#
#    1. YOLO count score  — normalised vehicle count
#    2. Occupancy score   — bbox area / lane polygon area
#    3. Motion score      — optical flow magnitude
#
#  Final score = w_yolo * yolo + w_occ * occ + w_mot * motion
#
#  Density label is derived from the raw YOLO count
#  (keeps label consistent with signal timing rules),
#  while the blended score is reported separately for
#  research/benchmarking.
# ─────────────────────────────────────────────────

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DENSITY_LOW_MAX, DENSITY_MEDIUM_MAX, LANE_PALETTE,
                    DENSITY_WEIGHT_YOLO, DENSITY_WEIGHT_OCCUPANCY,
                    DENSITY_WEIGHT_MOTION, LANE_ZONES)

# Max vehicles used to normalise YOLO count score to [0, 1]
MAX_VEHICLES_NORM = 25.0


class VehicleCounter:
    """
    Counts vehicles, computes hybrid density, annotates frame.

    count_and_annotate() now accepts optional:
      - lane_masks   : {lane_name: binary mask np.ndarray}
      - motion_scores: {lane_name: float in [0, 1]}
    """

    DENSITY_COLORS = {
        "Low":    (0, 255, 0),
        "Medium": (0, 165, 255),
        "High":   (0, 0, 255),
    }

    def __init__(self):
        self._lane_masks = {}
        self._mask_size  = None

    # ── Public API ──────────────────────────────

    @staticmethod
    def classify_density(count: int) -> str:
        """Map vehicle count → density label (uses YOLO count only)."""
        if count <= DENSITY_LOW_MAX:
            return "Low"
        elif count <= DENSITY_MEDIUM_MAX:
            return "Medium"
        else:
            return "High"

    def count_and_annotate(self,
                           frame,
                           lane_detections: dict,
                           motion_scores: dict = None) -> tuple:
        """
        Parameters
        ----------
        frame           : BGR numpy array (display resolution)
        lane_detections : {lane_name: [detection_dict, ...]}
        motion_scores   : {lane_name: float [0,1]} optical flow per lane

        Returns
        -------
        (annotated_frame, lane_stats)

        lane_stats = {
          lane_name: {
            "count":          int,
            "density":        str,        # Low / Medium / High
            "occupancy":      float,      # bbox area / lane area
            "motion":         float,      # optical flow score
            "hybrid_score":   float,      # blended score [0,1]
          }
        }
        """
        fh, fw = frame.shape[:2]
        motion_scores = motion_scores or {}

        # Build / refresh polygon masks for occupancy calculation
        if self._mask_size != (fw, fh):
            self._build_masks(fw, fh)
            self._mask_size = (fw, fh)

        lane_stats = {}
        lane_names = list(lane_detections.keys())

        for lane, detections in lane_detections.items():
            count   = len(detections)
            density = self.classify_density(count)

            # ── Occupancy score ──────────────────────────
            occ_score = 0.0
            lane_mask = self._lane_masks.get(lane)
            if lane_mask is not None:
                lane_area = int(lane_mask.sum())
                if lane_area > 0:
                    bbox_area = 0
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(fw, x2); y2 = min(fh, y2)
                        # Count only pixels inside the lane polygon
                        if x2 > x1 and y2 > y1:
                            patch = lane_mask[y1:y2, x1:x2]
                            bbox_area += int(patch.sum())
                    occ_score = min(1.0, bbox_area / lane_area)

            # ── Motion score ─────────────────────────────
            mot_score = float(motion_scores.get(lane, 0.0))

            # ── YOLO count score ─────────────────────────
            yolo_score = min(1.0, count / MAX_VEHICLES_NORM)

            # ── Hybrid blend ─────────────────────────────
            hybrid = (DENSITY_WEIGHT_YOLO      * yolo_score +
                      DENSITY_WEIGHT_OCCUPANCY  * occ_score  +
                      DENSITY_WEIGHT_MOTION     * mot_score)

            lane_stats[lane] = {
                "count":        count,
                "density":      density,
                "occupancy":    round(occ_score, 3),
                "motion":       round(mot_score, 3),
                "hybrid_score": round(hybrid,    3),
            }

            # ── Annotate frame ───────────────────────────
            lane_idx  = lane_names.index(lane) if lane in lane_names else 0
            box_color = self.DENSITY_COLORS[density]

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['conf']:.0%}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(frame,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              box_color, -1)
                cv2.putText(frame, label,
                            (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                            (255, 255, 255), 1)

            # Per-lane HUD overlay
            roi_color   = LANE_PALETTE[lane_idx % len(LANE_PALETTE)]
            hybrid_pct  = int(hybrid * 100)
            summary_text = f"{lane}: {count} | {density} | H:{hybrid_pct}%"
            cv2.putText(frame, summary_text,
                        (10, 30 + lane_names.index(lane) * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, roi_color, 2)

        return frame, lane_stats

    # ── Internal ────────────────────────────────

    def _build_masks(self, fw: int, fh: int) -> None:
        """Build binary polygon masks for each lane."""
        self._lane_masks = {}
        for lane, frac_pts in LANE_ZONES.items():
            mask = np.zeros((fh, fw), dtype=np.uint8)
            px = np.array([(int(x*fw), int(y*fh)) for x, y in frac_pts],
                          dtype=np.int32)
            cv2.fillPoly(mask, [px], 255)
            self._lane_masks[lane] = mask.astype(bool)
