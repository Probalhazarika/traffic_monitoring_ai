# ─────────────────────────────────────────────────────────────────────────────
#  lane_graph.py
#  Skeletonize binary lane masks → Lane graph → JSON export.
#
#  Pipeline
#  ────────
#  1. Binary mask  → skimage.morphology.skeletonize
#  2. Skeleton     → connected components  (cv2.connectedComponentsWithStats)
#  3. Each component → ordered polyline  (graph traversal from endpoint)
#  4. Polyline     → smoothed spline     (scipy.interpolate.splprep/splev)
#  5. Direction    → tangent angle at centroid
#  6. Output       → {lane_id: LaneData}  +  JSON export
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import cv2
from skimage.morphology import skeletonize


# ─────────────────────────────────────────────────────────────────────────────
#  Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LaneData:
    """Complete description of one detected lane centerline."""
    lane_id:    int
    polyline:   list[tuple[int, int]]    # [(x, y), ...] pixel coords
    spline_x:   list[float]              # smoothed X coords (finer resolution)
    spline_y:   list[float]              # smoothed Y coords
    direction:  float                    # dominant angle in degrees [0, 180)
    length:     float                    # arc length in pixels
    bbox:       tuple[int, int, int, int] # (x1, y1, x2, y2)
    pixel_count: int

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: chain skeleton pixels into ordered polyline
# ─────────────────────────────────────────────────────────────────────────────

def _chain_pixels(skel_mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a thin skeleton (binary mask) into an ordered list of (x, y) pixels.

    Strategy: find an endpoint (pixel with only 1 neighbour), then perform
    a DFS/BFS traversal.  Falls back to centroid-sorted path if no endpoint.
    """
    ys, xs = np.where(skel_mask > 0)
    if len(xs) == 0:
        return []

    pts_set = set(zip(xs.tolist(), ys.tolist()))
    pts_arr = list(pts_set)

    if len(pts_arr) == 1:
        return pts_arr

    # Build adjacency (8-connected)
    def neighbours(x, y):
        n = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in pts_set:
                    n.append((x + dx, y + dy))
        return n

    # Find endpoint (degree == 1) or start from leftmost point
    start = None
    for p in pts_arr:
        if len(neighbours(*p)) == 1:
            start = p
            break
    if start is None:
        start = min(pts_arr, key=lambda p: (p[0], p[1]))

    # DFS traversal
    ordered = []
    visited = set()
    stack   = [start]
    while stack:
        pt = stack.pop()
        if pt in visited:
            continue
        visited.add(pt)
        ordered.append(pt)
        for nb in neighbours(*pt):
            if nb not in visited:
                stack.append(nb)

    return ordered


# ─────────────────────────────────────────────────────────────────────────────
#  Spline fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_spline(pts: list[tuple[int, int]],
                n_out: int = 200,
                smooth: float = 5.0) -> tuple[list[float], list[float]]:
    """
    Fit a smoothing spline to an ordered list of (x, y) points.

    Returns (spline_x, spline_y) at n_out evenly-spaced parameter values.
    Falls back to linear interpolation if scipy fails.
    """
    if len(pts) < 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return xs, ys

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # Remove duplicate consecutive points
    mask = np.ones(len(xs), dtype=bool)
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1] and ys[i] == ys[i - 1]:
            mask[i] = False
    xs, ys = xs[mask], ys[mask]

    if len(xs) < 4:
        return xs.tolist(), ys.tolist()

    try:
        from scipy.interpolate import splprep, splev
        tck, u = splprep([xs, ys], s=smooth * len(xs), k=min(3, len(xs) - 1))
        u_fine  = np.linspace(0, 1, n_out)
        sx, sy  = splev(u_fine, tck)
        return sx.tolist(), sy.tolist()
    except Exception:
        # Fallback: resample linearly
        t  = np.linspace(0, 1, n_out)
        ti = np.linspace(0, 1, len(xs))
        return (np.interp(t, ti, xs).tolist(),
                np.interp(t, ti, ys).tolist())


# ─────────────────────────────────────────────────────────────────────────────
#  Direction from tangent
# ─────────────────────────────────────────────────────────────────────────────

def _lane_direction(pts: list[tuple[int, int]]) -> float:
    """
    Compute dominant lane direction (angle in degrees, [0°, 180°]).
    Uses linear regression on the point set.
    """
    if len(pts) < 2:
        return 0.0

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # PCA to find principal axis
    xs -= xs.mean(); ys -= ys.mean()
    cov = np.cov(np.stack([xs, ys]))
    _, vecs = np.linalg.eigh(cov)
    principal = vecs[:, -1]          # eigenvector of largest eigenvalue
    angle = math.degrees(math.atan2(principal[1], principal[0]))
    return angle % 180.0


# ─────────────────────────────────────────────────────────────────────────────
#  Arc length
# ─────────────────────────────────────────────────────────────────────────────

def _arc_length(pts: list[tuple[int, int]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        total += math.hypot(dx, dy)
    return total


# ─────────────────────────────────────────────────────────────────────────────
#  Main LaneGraph class
# ─────────────────────────────────────────────────────────────────────────────

class LaneGraph:
    """
    Convert a binary lane segmentation mask into a structured lane graph.

    Usage
    ─────
        lg    = LaneGraph(min_lane_length=50)
        lanes = lg.extract(binary_mask)
        lg.draw(frame, lanes)
        json_str = lg.to_json(lanes)
    """

    def __init__(
        self,
        min_lane_length:  int   = 50,    # minimum skeleton pixels to keep
        min_lane_area:    int   = 100,   # minimum mask pixels
        spline_points:    int   = 200,   # output spline resolution
        spline_smooth:    float = 5.0,
        morph_close_k:    int   = 5,     # pre-skeleton morphological close
        morph_open_k:     int   = 3,
    ):
        self.min_lane_length = min_lane_length
        self.min_lane_area   = min_lane_area
        self.spline_points   = spline_points
        self.spline_smooth   = spline_smooth
        self.morph_close_k   = morph_close_k
        self.morph_open_k    = morph_open_k

    # ── Core extraction ────────────────────────────────────────────────────

    def extract(self, binary_mask: np.ndarray) -> dict[int, LaneData]:
        """
        Extract lane graph from a binary mask.

        Parameters
        ----------
        binary_mask : (H, W) uint8, values in {0, 1} or {0, 255}

        Returns
        -------
        dict mapping lane_id → LaneData
        """
        # Binarise
        mask = (binary_mask > 0).astype(np.uint8)

        # Morphological pre-processing
        if self.morph_close_k > 0:
            k    = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (self.morph_close_k,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        if self.morph_open_k > 0:
            k    = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (self.morph_open_k,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        # Skeletonize
        skel = skeletonize(mask.astype(bool)).astype(np.uint8)

        # Connected components on skeleton
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            skel, connectivity=8
        )

        # Also run components on the original mask (for pixel_count / bbox)
        _, mask_labels, mask_stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        lanes: dict[int, LaneData] = {}
        lane_id = 1

        for label in range(1, n_labels):
            skel_pix = int(stats[label, cv2.CC_STAT_AREA])
            if skel_pix < self.min_lane_length:
                continue

            # Extract skeleton pixels for this label
            component_mask = (labels == label).astype(np.uint8)
            polyline = _chain_pixels(component_mask)

            if len(polyline) < 2:
                continue

            # Spline
            sx, sy = _fit_spline(polyline, self.spline_points, self.spline_smooth)

            # Direction + length
            direction = _lane_direction(polyline)
            length    = _arc_length(polyline)

            # Bounding box
            xs_ = [p[0] for p in polyline]
            ys_ = [p[1] for p in polyline]
            bbox = (min(xs_), min(ys_), max(xs_), max(ys_))

            lanes[lane_id] = LaneData(
                lane_id=lane_id,
                polyline=polyline,
                spline_x=sx,
                spline_y=sy,
                direction=direction,
                length=length,
                bbox=bbox,
                pixel_count=skel_pix,
            )
            lane_id += 1

        return lanes

    # ── Visualisation ─────────────────────────────────────────────────────

    COLORS = [
        (0,   255,   0),   # green
        (255,  165,  0),   # orange
        (0,   165, 255),   # blue
        (255,   0, 255),   # magenta
        (0,   255, 255),   # cyan
        (255, 255,   0),   # yellow
        (255,  80,  80),   # salmon
        (80,  255,  80),   # lime
    ]

    def draw(
        self,
        frame:     np.ndarray,
        lanes:     dict[int, LaneData],
        draw_spline:  bool = True,
        draw_id:      bool = True,
        draw_bbox:    bool = False,
        thickness:    int  = 2,
    ) -> np.ndarray:
        """Overlay lane centerlines and IDs onto frame (in-place copy)."""
        out = frame.copy()

        for lid, lane in lanes.items():
            color = self.COLORS[(lid - 1) % len(self.COLORS)]

            if draw_spline and len(lane.spline_x) > 1:
                pts = np.array(
                    list(zip([int(x) for x in lane.spline_x],
                              [int(y) for y in lane.spline_y])),
                    dtype=np.int32
                )
                cv2.polylines(out, [pts], isClosed=False,
                              color=color, thickness=thickness, lineType=cv2.LINE_AA)
            else:
                # Fallback: draw raw polyline
                pts = np.array(lane.polyline, dtype=np.int32)
                cv2.polylines(out, [pts], isClosed=False,
                              color=color, thickness=thickness)

            if draw_id and len(lane.spline_x) > 0:
                mid = len(lane.spline_x) // 2
                cx, cy = int(lane.spline_x[mid]), int(lane.spline_y[mid])
                cv2.putText(out, f"L{lid}",
                            (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2, cv2.LINE_AA)

            if draw_bbox:
                x1, y1, x2, y2 = lane.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        return out

    # ── JSON export ───────────────────────────────────────────────────────

    @staticmethod
    def to_json(lanes: dict[int, LaneData], indent: int = 2) -> str:
        """Export lane graph as a JSON string."""
        data = {
            "lanes": {
                str(lid): {
                    "lane_id":    lane.lane_id,
                    "direction":  round(lane.direction, 2),
                    "length":     round(lane.length, 2),
                    "pixel_count": lane.pixel_count,
                    "bbox":       list(lane.bbox),
                    "polyline":   [[int(x), int(y)] for x, y in lane.polyline],
                    "spline":     [[round(x, 1), round(y, 1)]
                                   for x, y in zip(lane.spline_x, lane.spline_y)],
                }
                for lid, lane in lanes.items()
            },
            "num_lanes": len(lanes),
        }
        return json.dumps(data, indent=indent)

    @staticmethod
    def save_json(lanes: dict[int, LaneData], path: str) -> None:
        """Save lane graph JSON to disk."""
        with open(path, "w") as f:
            f.write(LaneGraph.to_json(lanes))
        print(f"[LaneGraph] JSON saved → {path}")

    @staticmethod
    def load_json(path: str) -> dict:
        """Load lane graph from JSON file."""
        with open(path) as f:
            return json.load(f)

    # ── Nearest lane query (for vehicle assignment) ───────────────────────

    @staticmethod
    def nearest_lane(
        point:  tuple[int, int],
        lanes:  dict[int, LaneData],
        max_dist: float = 200.0,
    ) -> tuple[Optional[int], float]:
        """
        Find the lane ID whose spline is closest to the given (x, y) point.

        Returns (lane_id, distance) or (None, inf) if no lane within max_dist.
        """
        px, py = point
        best_id   = None
        best_dist = float("inf")

        for lid, lane in lanes.items():
            if not lane.spline_x:
                continue
            sx = np.array(lane.spline_x)
            sy = np.array(lane.spline_y)
            dists = np.hypot(sx - px, sy - py)
            d     = float(np.min(dists))
            if d < best_dist:
                best_dist = d
                best_id   = lid

        if best_dist > max_dist:
            return None, best_dist
        return best_id, best_dist
