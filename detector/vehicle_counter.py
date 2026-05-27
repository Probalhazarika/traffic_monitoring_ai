# ─────────────────────────────────────────────────
#  detector/vehicle_counter.py
#  Counts vehicles per lane and classifies density.
# ─────────────────────────────────────────────────

import cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DENSITY_LOW_MAX, DENSITY_MEDIUM_MAX, LANE_PALETTE


class VehicleCounter:
    """
    Given lane→detections mapping, produces:
      - vehicle count per lane
      - density label per lane  (Low / Medium / High)
    Also annotates the frame with bounding boxes and counts.
    """

    # Density colour codes for bounding boxes (BGR)
    DENSITY_COLORS = {
        "Low":    (0, 255, 0),      # green
        "Medium": (0, 165, 255),    # orange
        "High":   (0, 0, 255),      # red
    }

    # ── Public API ──────────────────────────────

    @staticmethod
    def classify_density(count: int) -> str:
        """Map vehicle count → density label."""
        if count <= DENSITY_LOW_MAX:
            return "Low"
        elif count <= DENSITY_MEDIUM_MAX:
            return "Medium"
        else:
            return "High"

    def count_and_annotate(self, frame, lane_detections: dict) -> tuple:
        """
        Parameters
        ----------
        frame           : BGR numpy array
        lane_detections : {lane_name: [detection_dict, ...]}

        Returns
        -------
        annotated_frame : frame with boxes + counts drawn
        lane_stats      : {lane_name: {"count": int, "density": str}}
        """
        lane_stats = {}

        for lane, detections in lane_detections.items():
            count   = len(detections)
            density = self.classify_density(count)
            lane_stats[lane] = {"count": count, "density": density}

            # Pick color from palette by lane index (fallback to white)
            lane_names = list(lane_detections.keys())
            lane_idx   = lane_names.index(lane) if lane in lane_names else 0
            roi_color  = LANE_PALETTE[lane_idx % len(LANE_PALETTE)]
            box_color = self.DENSITY_COLORS[density]

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['label']} {det['conf']:.0%}"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Label background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame,
                              (x1, y1 - th - 8), (x1 + tw + 4, y1),
                              box_color, -1)
                cv2.putText(frame, label,
                            (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)

            # Per-lane count overlay
            summary_text = f"{lane}: {count} | {density}"
            cv2.putText(frame, summary_text,
                        (10, 30 + list(lane_detections.keys()).index(lane) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

        return frame, lane_stats
