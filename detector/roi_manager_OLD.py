# ─────────────────────────────────────────────────
#  detector/roi_manager.py
#  Manages lane ROI regions and assigns detections
#  to the correct lane.
# ─────────────────────────────────────────────────

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANE_ROIS, ROI_COLORS


class ROIManager:
    """
    Handles rectangular Region-Of-Interest (ROI) zones for each lane.

    A vehicle is considered 'in' a lane when its bounding-box centre (cx, cy)
    falls inside that lane's ROI rectangle.
    """

    def __init__(self, rois: dict = None, frame_size: tuple = None):
        """
        Parameters
        ----------
        rois : dict  {lane_name: (x1, y1, x2, y2)}
                     If None, uses values from config.py
        frame_size : (width, height) – used to scale ROIs proportionally
                     when the actual video resolution differs from 1280×720.
                     Pass None to skip scaling.
        """
        self.rois   = rois if rois is not None else dict(LANE_ROIS)
        self.colors = dict(ROI_COLORS)

        # Scale ROIs if the actual frame size differs from the default 1280×720
        if frame_size is not None:
            w, h = frame_size
            sx, sy = w / 1280, h / 720
            self.rois = {
                lane: (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                for lane, (x1, y1, x2, y2) in self.rois.items()
            }

    # ── Public API ──────────────────────────────

    def assign_to_lanes(self, detections: list) -> dict:
        """
        Map each detection to a lane based on bbox centre.

        Returns
        -------
        dict  {lane_name: [detection, ...]}
        """
        lane_detections = {lane: [] for lane in self.rois}

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            for lane, (x1, y1, x2, y2) in self.rois.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    lane_detections[lane].append(det)
                    break   # each vehicle counted in one lane only

        return lane_detections

    def draw_rois(self, frame):
        """
        Draw coloured ROI rectangles and lane labels onto a frame (in-place).
        """
        for lane, (x1, y1, x2, y2) in self.rois.items():
            color = self.colors.get(lane, (255, 255, 255))
            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label
            cv2.putText(frame, lane, (x1 + 8, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    def get_roi_for_lane(self, lane_name: str):
        """Return (x1, y1, x2, y2) for the given lane."""
        return self.rois.get(lane_name)
