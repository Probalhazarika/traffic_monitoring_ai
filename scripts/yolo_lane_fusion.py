# ─────────────────────────────────────────────────────────────────────────────
#  yolo_lane_fusion.py
#  Vehicle-to-lane assignment: fuse YOLO detections with lane graph.
#
#  Assignment strategy
#  ───────────────────
#  For each detected vehicle (bounding box → centre point):
#    1. Find the nearest lane spline (Euclidean distance to all spline points)
#    2. Assign vehicle to that lane if distance < max_dist
#    3. Optionally confirm via point-in-polygon (mask region) test
#
#  Output
#  ──────
#  - Per-frame list of {frame_id, vehicle_id, lane_id, x, y}
#  - Running CSV export
#  - Per-lane vehicle counts + density estimates
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from lane_graph import LaneData, LaneGraph


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleLaneAssignment:
    """Single vehicle-lane assignment record."""
    frame_id:   int
    vehicle_id: int
    lane_id:    Optional[int]    # None if no lane within max_dist
    x:          int              # vehicle centre X (display coords)
    y:          int              # vehicle centre Y
    dist:       float            # distance to nearest lane centrepoint
    confidence: float            # detection confidence
    cls_label:  str              # 'car', 'truck', etc.

    def to_row(self) -> list:
        return [self.frame_id, self.vehicle_id, self.lane_id or -1,
                self.x, self.y, round(self.dist, 1),
                round(self.confidence, 3), self.cls_label]


# ─────────────────────────────────────────────────────────────────────────────
#  Density levels
# ─────────────────────────────────────────────────────────────────────────────

def vehicle_count_to_density(count: int,
                              low_max: int = 3,
                              med_max: int = 7) -> str:
    if count <= low_max:  return "Low"
    if count <= med_max:  return "Medium"
    return "High"


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion class
# ─────────────────────────────────────────────────────────────────────────────

class YOLOLaneFusion:
    """
    Assign YOLO vehicle detections to detected lane centerlines.

    Integrates with the existing YOLODetector output format:
      detection = {
          "bbox":   (x1, y1, x2, y2),
          "cls_id": int,
          "label":  str,
          "conf":   float,
          "cx":     int,
          "cy":     int,
      }

    Parameters
    ----------
    max_dist      : max pixel distance to assign a vehicle to a lane
    use_bottom    : use bottom-centre of bbox (more grounded) vs true centre
    output_csv    : path to write running CSV log (None = skip)
    density_low   : vehicle count threshold for "Low" density
    density_med   : vehicle count threshold for "Medium" density
    """

    CSV_HEADER = ["frame_id", "vehicle_id", "lane_id", "x", "y",
                  "dist", "confidence", "class"]

    def __init__(
        self,
        max_dist:    float = 120.0,
        use_bottom:  bool  = True,
        output_csv:  Optional[str] = "outputs/vehicle_lane_assignments.csv",
        density_low: int   = 3,
        density_med: int   = 7,
    ):
        self.max_dist    = max_dist
        self.use_bottom  = use_bottom
        self.density_low = density_low
        self.density_med = density_med

        # Running history
        self._assignments: list[VehicleLaneAssignment] = []
        self._lane_counts:  dict[int, int] = defaultdict(int)
        self._frame_history: list[dict]    = []

        # CSV writer
        self._csv_file   = None
        self._csv_writer = None
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            self._csv_file   = open(output_csv, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.CSV_HEADER)
            print(f"[YOLOLaneFusion] CSV → {output_csv}")

    # ── Main API ───────────────────────────────────────────────────────────

    def assign(
        self,
        detections: list[dict],
        lanes:      dict[int, LaneData],
        frame_id:   int = 0,
    ) -> list[VehicleLaneAssignment]:
        """
        Assign every detection to the nearest lane.

        Parameters
        ----------
        detections : list of YOLO detection dicts
        lanes      : dict from LaneGraph.extract() or LaneTracker.update()
        frame_id   : current frame index

        Returns
        -------
        List of VehicleLaneAssignment records for this frame.
        """
        frame_assignments = []
        frame_lane_counts: dict[Optional[int], int] = defaultdict(int)

        for vid, det in enumerate(detections):
            bbox  = det.get("bbox", (0, 0, 0, 0))
            cx    = det.get("cx", (bbox[0] + bbox[2]) // 2)
            cy    = det.get("cy", (bbox[1] + bbox[3]) // 2)

            if self.use_bottom:
                # Bottom-centre of bbox → closer to road surface
                bx = (bbox[0] + bbox[2]) // 2
                by = bbox[3]
            else:
                bx, by = cx, cy

            lane_id, dist = LaneGraph.nearest_lane(
                (bx, by), lanes, max_dist=self.max_dist
            )

            record = VehicleLaneAssignment(
                frame_id=frame_id,
                vehicle_id=vid,
                lane_id=lane_id,
                x=bx,
                y=by,
                dist=dist,
                confidence=det.get("conf", 0.0),
                cls_label=det.get("label", "vehicle"),
            )
            frame_assignments.append(record)
            if lane_id is not None:
                frame_lane_counts[lane_id] += 1

        # Update running counts
        for lid, cnt in frame_lane_counts.items():
            self._lane_counts[lid] += cnt

        self._assignments.extend(frame_assignments)
        self._frame_history.append({
            "frame_id":    frame_id,
            "assignments": [a.to_row() for a in frame_assignments],
            "lane_counts": dict(frame_lane_counts),
        })

        # Write to CSV
        if self._csv_writer:
            for a in frame_assignments:
                self._csv_writer.writerow(a.to_row())

        return frame_assignments

    # ── Lane density summary ───────────────────────────────────────────────

    def lane_density(
        self,
        frame_assignments: list[VehicleLaneAssignment],
        lanes: dict[int, LaneData],
    ) -> dict[int, dict]:
        """
        Compute per-lane density for the current frame.

        Returns
        -------
        dict lane_id → {count, density_label, lane_data}
        """
        counts: dict[int, int] = defaultdict(int)
        for a in frame_assignments:
            if a.lane_id is not None:
                counts[a.lane_id] += 1

        result = {}
        for lid, lane in lanes.items():
            cnt   = counts.get(lid, 0)
            label = vehicle_count_to_density(cnt, self.density_low, self.density_med)
            result[lid] = {
                "count":   cnt,
                "density": label,
                "lane_data": lane,
            }
        return result

    def draw_assignments(
        self,
        frame:       np.ndarray,
        assignments: list[VehicleLaneAssignment],
        detections:  list[dict],
        lanes:       dict[int, LaneData],
    ) -> np.ndarray:
        """
        Draw vehicle bboxes coloured by lane assignment.
        Each vehicle label shows "L{lane_id}" or "?" if unassigned.
        """
        import cv2

        LANE_COLORS = [
            (0, 255, 0), (255, 165, 0), (0, 165, 255),
            (255, 0, 255), (0, 255, 255), (255, 255, 0),
        ]
        out = frame.copy()

        for a, det in zip(assignments, detections):
            bbox = det.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            lid   = a.lane_id
            color = LANE_COLORS[(lid - 1) % len(LANE_COLORS)] if lid else (128, 128, 128)
            label = f"L{lid} {det.get('label','')}" if lid else f"? {det.get('label','')}"
            conf  = f"{det.get('conf', 0):.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{label} {conf}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

        return out

    # ── Heatmap ────────────────────────────────────────────────────────────

    def build_density_heatmap(
        self,
        frame_h: int,
        frame_w: int,
        window:  int = 30,
        sigma:   float = 20.0,
    ) -> np.ndarray:
        """
        Build a Gaussian density heatmap from the last `window` frames.
        Returns (H, W, 3) BGR heatmap.
        """
        import cv2

        hmap = np.zeros((frame_h, frame_w), dtype=np.float32)
        recent = self._frame_history[-window:]

        for frame_data in recent:
            for row in frame_data["assignments"]:
                _, _, lid, x, y = row[:5]
                if lid > 0 and 0 <= x < frame_w and 0 <= y < frame_h:
                    hmap[y, x] += 1.0

        # Gaussian blur
        k  = max(1, int(sigma * 3) | 1)
        hmap = cv2.GaussianBlur(hmap, (k, k), sigma)

        # Normalise and colorize
        if hmap.max() > 0:
            hmap = (hmap / hmap.max() * 255).astype(np.uint8)
        else:
            hmap = hmap.astype(np.uint8)

        return cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    # ── Cleanup ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def __del__(self):
        self.close()

    # ── Stats ──────────────────────────────────────────────────────────────

    @property
    def total_assignments(self) -> int:
        return len(self._assignments)

    def per_lane_total(self) -> dict[int, int]:
        return dict(self._lane_counts)

    def summary(self) -> str:
        lines = ["[YOLOLaneFusion] Summary:"]
        for lid, cnt in sorted(self._lane_counts.items()):
            lines.append(f"  Lane {lid}: {cnt} vehicle-frame detections")
        lines.append(f"  Total: {self.total_assignments} assignments")
        return "\n".join(lines)
