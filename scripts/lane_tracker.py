# ─────────────────────────────────────────────────────────────────────────────
#  lane_tracker.py
#  Temporal lane tracking with exponential moving average (EMA) smoothing.
#
#  Problem: per-frame segmentation is noisy — lane splines jump between frames.
#  Solution: maintain a persistent lane track across time with IoU-based
#            association and EMA smoothing on spline control points.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from lane_graph import LaneData, LaneGraph, _lane_direction


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent lane track
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LaneTrack:
    """A persistent lane identity tracked across frames."""
    track_id:    int
    last_seen:   int                    # frame index
    age:         int      = 0           # frames the track has existed
    hits:        int      = 0           # frames with a detection
    lane_data:   Optional[LaneData] = None

    # EMA smoothed spline (lists of float)
    smooth_x:    list[float] = field(default_factory=list)
    smooth_y:    list[float] = field(default_factory=list)

    # Density history
    density_hist: list[int] = field(default_factory=list)

    def is_confirmed(self, min_hits: int = 3) -> bool:
        return self.hits >= min_hits

    def occupancy(self, frame_w: int, frame_h: int, lane_zones: dict) -> float:
        """Fraction of lane zone covered by detections (placeholder)."""
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Association helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_overlap(b1: tuple, b2: tuple) -> float:
    """Intersection over Union for two bounding boxes (x1,y1,x2,y2)."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _direction_diff(a: float, b: float) -> float:
    """Angular difference in [0°, 90°] range (direction-invariant)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _centroid_distance(lane1: LaneData, lane2: LaneData) -> float:
    """Euclidean distance between midpoints of two lane splines."""
    def mid(l):
        n = len(l.spline_x)
        if n == 0:
            xs = [p[0] for p in l.polyline]
            ys = [p[1] for p in l.polyline]
            return np.mean(xs) if xs else 0, np.mean(ys) if ys else 0
        return l.spline_x[n // 2], l.spline_y[n // 2]

    cx1, cy1 = mid(lane1)
    cx2, cy2 = mid(lane2)
    return math.hypot(cx1 - cx2, cy1 - cy2)


# ─────────────────────────────────────────────────────────────────────────────
#  Lane Tracker
# ─────────────────────────────────────────────────────────────────────────────

class LaneTracker:
    """
    Multi-frame lane tracker using greedy association and EMA smoothing.

    Algorithm per frame
    ───────────────────
    1. Extract new lane detections from LaneGraph.
    2. Associate detections to existing tracks via centroid distance + direction.
    3. Update matched tracks: EMA smooth spline, increment hits.
    4. Unmatched detections → new tentative tracks.
    5. Missed tracks: increment miss counter; prune if missed > max_age.
    6. Return confirmed tracks as stabilised lanes.

    Parameters
    ----------
    ema_alpha      : EMA weight for new detection (1-alpha = old smoothed value)
    max_age        : frames a track survives without a match
    min_hits       : detections needed before a track is "confirmed"
    max_centroid_dist : pixel distance threshold for association
    max_dir_diff   : angular difference threshold (degrees)
    spline_points  : resolution for EMA-smoothed spline output
    """

    def __init__(
        self,
        ema_alpha:         float = 0.35,
        max_age:           int   = 5,
        min_hits:          int   = 3,
        max_centroid_dist: float = 150.0,
        max_dir_diff:      float = 30.0,
        spline_points:     int   = 200,
    ):
        self.ema_alpha         = ema_alpha
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.max_centroid_dist = max_centroid_dist
        self.max_dir_diff      = max_dir_diff
        self.spline_points     = spline_points

        self._tracks: dict[int, LaneTrack] = {}
        self._next_id = 1
        self._frame   = 0

        # History: frame_id → {track_id: LaneData}
        self.history: list[dict] = []

        # Per-frame density: track_id → vehicle count
        self.density: dict[int, int] = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, detections: dict[int, LaneData]) -> dict[int, LaneData]:
        """
        Update tracks with new frame detections.

        Parameters
        ----------
        detections : dict returned by LaneGraph.extract()

        Returns
        -------
        dict[track_id → smoothed LaneData] for confirmed tracks only.
        """
        self._frame += 1
        det_list = list(detections.values())

        # ── Association ───────────────────────────────────────────────────
        unmatched_dets   = list(range(len(det_list)))
        unmatched_tracks = list(self._tracks.keys())
        matches          = []

        if det_list and unmatched_tracks:
            # Cost matrix: centroid distance (penalise direction mismatch)
            cost = np.full((len(det_list), len(unmatched_tracks)), 1e9)
            for di, d in enumerate(det_list):
                for ti, tid in enumerate(unmatched_tracks):
                    t = self._tracks[tid]
                    if t.lane_data is None:
                        continue
                    cdist  = _centroid_distance(d, t.lane_data)
                    ddiff  = _direction_diff(d.direction, t.lane_data.direction)
                    if (cdist <= self.max_centroid_dist and
                            ddiff <= self.max_dir_diff):
                        cost[di, ti] = cdist + ddiff * 2.0

            # Greedy assignment (Hungarian is overkill at this scale)
            while True:
                if cost.size == 0 or cost.min() > 1e8:
                    break
                di, ti = np.unravel_index(np.argmin(cost), cost.shape)
                tid     = unmatched_tracks[ti]
                matches.append((di, tid))
                cost[di, :] = 1e9
                cost[:, ti] = 1e9
                unmatched_dets.remove(di)
                unmatched_tracks.remove(tid)

        # ── Update matched tracks ─────────────────────────────────────────
        for di, tid in matches:
            det   = det_list[di]
            track = self._tracks[tid]
            track.hits     += 1
            track.last_seen = self._frame
            track.age      += 1

            # EMA smooth spline
            if not track.smooth_x or len(track.smooth_x) != len(det.spline_x):
                track.smooth_x = list(det.spline_x)
                track.smooth_y = list(det.spline_y)
            else:
                n  = min(len(track.smooth_x), len(det.spline_x))
                sx = [self.ema_alpha * det.spline_x[i] +
                      (1 - self.ema_alpha) * track.smooth_x[i]
                      for i in range(n)]
                sy = [self.ema_alpha * det.spline_y[i] +
                      (1 - self.ema_alpha) * track.smooth_y[i]
                      for i in range(n)]
                track.smooth_x = sx
                track.smooth_y = sy

            # Update lane_data direction (smooth)
            old_dir = track.lane_data.direction if track.lane_data else det.direction
            smooth_dir = (1 - self.ema_alpha) * old_dir + self.ema_alpha * det.direction
            track.lane_data = LaneData(
                lane_id=tid,
                polyline=det.polyline,
                spline_x=track.smooth_x,
                spline_y=track.smooth_y,
                direction=smooth_dir,
                length=det.length,
                bbox=det.bbox,
                pixel_count=det.pixel_count,
            )

        # ── Create new tracks for unmatched detections ────────────────────
        for di in unmatched_dets:
            det     = det_list[di]
            new_id  = self._next_id
            self._next_id += 1
            self._tracks[new_id] = LaneTrack(
                track_id=new_id,
                last_seen=self._frame,
                hits=1,
                age=1,
                lane_data=LaneData(
                    lane_id=new_id,
                    polyline=det.polyline,
                    spline_x=det.spline_x,
                    spline_y=det.spline_y,
                    direction=det.direction,
                    length=det.length,
                    bbox=det.bbox,
                    pixel_count=det.pixel_count,
                ),
                smooth_x=list(det.spline_x),
                smooth_y=list(det.spline_y),
            )

        # ── Prune old tracks ──────────────────────────────────────────────
        to_remove = [
            tid for tid, t in self._tracks.items()
            if self._frame - t.last_seen > self.max_age
        ]
        for tid in to_remove:
            del self._tracks[tid]

        # ── Return confirmed tracks ───────────────────────────────────────
        confirmed: dict[int, LaneData] = {}
        for tid, track in self._tracks.items():
            if track.is_confirmed(self.min_hits) and track.lane_data is not None:
                confirmed[tid] = track.lane_data

        self.history.append({tid: ld for tid, ld in confirmed.items()})
        return confirmed

    def reset(self) -> None:
        """Clear all tracks (e.g., when processing a new video)."""
        self._tracks.clear()
        self._next_id = 1
        self._frame   = 0
        self.history.clear()
        self.density.clear()

    def update_vehicle_count(self, track_id: int, count: int) -> None:
        """Register a vehicle count for a specific lane track (for density)."""
        self._tracks[track_id].density_hist.append(count) if track_id in self._tracks else None

    def confirmed_tracks(self) -> dict[int, LaneTrack]:
        return {tid: t for tid, t in self._tracks.items()
                if t.is_confirmed(self.min_hits)}

    @property
    def num_active(self) -> int:
        return len(self._tracks)

    @property
    def num_confirmed(self) -> int:
        return sum(1 for t in self._tracks.values()
                   if t.is_confirmed(self.min_hits))
