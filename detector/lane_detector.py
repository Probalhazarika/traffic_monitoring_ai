# ─────────────────────────────────────────────────────────────────
#  detector/lane_detector.py
#
#  Lane zone manager — loads hard-coded manual coordinates from
#  config.LANE_ZONES and exposes the assignment + drawing API.
#
#  No auto-detection, no Hough lines, no ML model.
#  Zones are the manually-calibrated fractional polygons set in
#  config.py for the specific drone footage.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANE_PALETTE, LANE_ZONES, FREE_TURN_ZONES


class LaneDetector:
    """
    Manages the manual lane zones defined in config.LANE_ZONES.

    Public API (used by processor.py)
    ──────────────────────────────────
    calibrate(frames, video_path)          → loads zones from config
    draw_lanes(frame, lane_stats)          → draws coloured polygons
    assign_vehicles_to_lanes(dets, w, h)   → returns {lane: [det,...]}
    get_polygons_serializable()            → plain-list form (for JSON)
    _refresh_polys(w, h)                   → rebuilds pixel coords on resize
    """

    def __init__(self):
        # Fractional coords  {lane_name: [(xf, yf), ...]}
        self._lane_fracs: dict       = {}
        self._free_fracs: dict       = {}
        # Pixel polygons — rebuilt whenever frame size changes
        self.lane_polygons:  dict    = {}
        self.free_turn_polygons: dict = {}
        # Pre-filled uint8 masks for fast bbox-overlap tests
        self._lane_masks:  dict      = {}
        self.calibrated:   bool      = False
        # Last frame size used to build pixel polygons
        self._last_w: int            = 0
        self._last_h: int            = 0

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def calibrate(self, frames: list = None, video_path: str = "") -> bool:
        """
        Load manual lane zones from config.LANE_ZONES.
        The `frames` argument is accepted for API compatibility but ignored.
        """
        if not LANE_ZONES:
            print("[LaneDetector] ✗ LANE_ZONES is empty in config.py — "
                  "please add manual coordinates.")
            return False

        self._lane_fracs = dict(LANE_ZONES)
        self._free_fracs = dict(FREE_TURN_ZONES) if FREE_TURN_ZONES else {}
        self.calibrated  = True

        print(f"[LaneDetector] ✓ Manual zones loaded: "
              f"{list(self._lane_fracs.keys())}")
        return True

    # ── Vehicle assignment ────────────────────────────────────────

    def assign_vehicles_to_lanes(self, detections: list,
                                  frame_w: int = 1280,
                                  frame_h: int = 720) -> dict:
        """
        Assign each detection to a lane via bounding-box overlap.

        A vehicle is assigned to a lane when ≥ 10 % of its bounding-box
        area falls inside that lane's polygon mask.

        Returns {lane_name: [det, ...]}
        """
        OVERLAP_THRESHOLD = 0.10

        result = {name: [] for name in self._lane_fracs}

        if not self.calibrated:
            return result

        self._refresh_polys(frame_w, frame_h)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            for lane, mask in self._lane_masks.items():
                if mask is None:
                    continue

                ph, pw = mask.shape[:2]
                bx1 = max(0, x1);  by1 = max(0, y1)
                bx2 = min(pw, x2); by2 = min(ph, y2)
                if bx2 <= bx1 or by2 <= by1:
                    continue

                box_area     = (bx2 - bx1) * (by2 - by1)
                overlap_area = int(np.sum(mask[by1:by2, bx1:bx2] > 0))
                overlap_frac = overlap_area / box_area if box_area > 0 else 0.0

                if overlap_frac >= OVERLAP_THRESHOLD:
                    result[lane].append(det)
                    break   # first matching lane wins

        return result

    # ── Drawing ───────────────────────────────────────────────────

    def draw_lanes(self, frame: np.ndarray,
                   lane_stats: dict = None) -> np.ndarray:
        """
        Draw coloured, semi-transparent polygons for each lane zone.
        Optionally overlays vehicle count and density label.
        """
        h, w = frame.shape[:2]
        self._refresh_polys(w, h)

        for i, (lane, poly) in enumerate(self.lane_polygons.items()):
            if poly is None or len(poly) == 0:
                continue

            color = LANE_PALETTE[i % len(LANE_PALETTE)]

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

            # Bold border
            cv2.polylines(frame, [poly], isClosed=True,
                          color=color, thickness=4)

            # Centroid label
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if lane_stats and lane in lane_stats:
                count   = lane_stats[lane]["count"]
                density = lane_stats[lane]["density"]
                label   = f"{lane}: {count} cars [{density}]"
            else:
                label = lane

            (sw, sh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
            )
            cv2.rectangle(frame,
                          (cx - sw // 2 - 6, cy - sh - 6),
                          (cx + sw // 2 + 6, cy + 6),
                          (0, 0, 0), -1)
            cv2.putText(frame, label,
                        (cx - sw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        color, 2, cv2.LINE_AA)

        return frame

    # ── Serialisation ─────────────────────────────────────────────

    def get_polygons_serializable(self) -> dict:
        """Return lane polygons as plain Python lists (JSON-serialisable)."""
        return {
            lane: poly.tolist() if poly is not None else []
            for lane, poly in self.lane_polygons.items()
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _refresh_polys(self, w: int, h: int) -> None:
        """
        Rebuild pixel polygons and pre-filled masks from fractional coords
        whenever the frame dimensions change.
        """
        if (w == self._last_w and h == self._last_h
                and self.lane_polygons and self._lane_masks):
            return  # already up to date

        self._last_w, self._last_h = w, h

        # Main lane polygons + overlap masks
        self.lane_polygons = {}
        self._lane_masks   = {}
        for name, frac_pts in self._lane_fracs.items():
            pts = np.array(
                [[int(xf * w), int(yf * h)] for xf, yf in frac_pts],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            self.lane_polygons[name] = pts

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            self._lane_masks[name] = mask

        # Free-turn zones (display only)
        self.free_turn_polygons = {}
        for name, frac_pts in self._free_fracs.items():
            pts = np.array(
                [[int(xf * w), int(yf * h)] for xf, yf in frac_pts],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            self.free_turn_polygons[name] = pts
