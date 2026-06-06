# ─────────────────────────────────────────────────────────────────
#  detector/lane_detector.py
#
#  Classical CV pipeline for dynamic lane detection.
#
#  Pipeline per frame:
#    BGR frame
#    → resize to 1920×1080 for fast processing
#    → grayscale → Gaussian blur → Canny edges
#    → road-area mask (from config.ROAD_MASK_POLY)
#    → HoughLinesP → angle + proximity clustering
#    → extended lane-boundary lines
#    → adjacent boundary pairs → lane polygons
#
#  Calibration:
#    Call calibrate(frames) with the first N frames.
#    The detector accumulates all Hough detections across
#    frames, clusters them, and stores stable lane polygons
#    in self.lane_polygons {lane_name: np.ndarray(N,1,2)}.
#
#  Vehicle assignment:
#    assign_vehicles_to_lanes(detections) uses
#    cv2.pointPolygonTest() — accurate even for irregular polygons.
#    Vehicles outside every polygon go to the nearest lane.
#
#  Fallback:
#    If Hough finds fewer than 2 distinct boundaries, the road
#    mask is split into EXPECTED_LANES equal vertical strips.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CALIBRATION_FRAMES,
    CANNY_LOW, CANNY_HIGH,
    BLUR_KERNEL,
    HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP,
    LANE_ANGLE_MIN, LANE_ANGLE_MAX,
    LINE_CLUSTER_DIST, LINE_CLUSTER_ANGLE_DIFF,
    MAX_LANES, EXPECTED_LANES,
    ROAD_MASK_POLY, LANE_PALETTE, LANE_ZONES, FREE_TURN_ZONES,
)

# AUTO_LANE_VIDEO is written to config.py after the first auto-detection.
# It may not exist on a fresh install — handled gracefully below.
try:
    from config import AUTO_LANE_VIDEO as _AUTO_LANE_VIDEO
except ImportError:
    _AUTO_LANE_VIDEO = ""

# Resolution used internally for CV processing (fast, still detailed enough)
_CV_W, _CV_H = 1920, 1080


class LaneDetector:
    """
    Detects road lane regions using classical computer vision and assigns
    YOLO-detected vehicles to those regions.
    """

    def __init__(self):
        # Fractional coords {lane_name: [(xf, yf), ...]} — source of truth
        self._lane_fracs: dict      = {}
        self._free_fracs: dict      = {}
        # Pixel polygons — rebuilt per frame from actual frame size
        self.lane_polygons: dict    = {}
        self.free_turn_polygons: dict = {}
        # Pre-computed uint8 masks for fast overlap tests (same size as frame)
        self._lane_masks: dict      = {}
        self.calibrated: bool       = False
        self._orig_w: int           = 0
        self._orig_h: int           = 0
        # Last frame size used to build pixel polygons
        self._last_w: int           = 0
        self._last_h: int           = 0

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def calibrate(self, frames: list, video_path: str = "") -> bool:
        """
        Analyse the first N frames to locate stable lane boundaries.

        Priority order
        --------------
        1. LANE_ZONES set in config.py AND video matches → use stored zones.
           (Covers both hand-crafted zones and previously auto-detected+saved zones.)
        2. LANE_ZONES empty OR a different video is loaded → run AutoLaneDetector
           (the new computer-vision pipeline).  Detected zones are saved back to
           config.py so the next launch skips detection entirely.
        3. AutoLaneDetector fails → fall back to the legacy Hough-line detector.

        Parameters
        ----------
        frames     : list of BGR numpy arrays (raw, full-resolution)
        video_path : path to the current video file (used for cache validation)

        Returns
        -------
        True if at least one lane polygon was built, False otherwise.
        """
        if not frames:
            return False

        self._orig_h, self._orig_w = frames[0].shape[:2]

        # Determine whether the stored zones belong to this video
        video_basename = os.path.basename(video_path) if video_path else ""
        video_changed  = (bool(_AUTO_LANE_VIDEO)
                          and bool(video_basename)
                          and video_basename != _AUTO_LANE_VIDEO)

        if video_changed:
            print(f"[LaneDetector] ⚠  New video detected: '{video_basename}' "
                  f"(stored zones are for '{_AUTO_LANE_VIDEO}') — re-detecting.")

        effective_zones = {} if video_changed else LANE_ZONES

        # ── Priority 1: Valid stored zones ─────────────────────────
        if effective_zones:
            self._lane_fracs = dict(effective_zones)
            self._free_fracs = dict(FREE_TURN_ZONES) if FREE_TURN_ZONES else {}
            self.calibrated  = True
            n = len(self._lane_fracs)
            print(f"[LaneDetector] ✓ Using {'cached auto' if _AUTO_LANE_VIDEO else 'manual'} "
                  f"zones: {list(self._lane_fracs.keys())}")
            return n >= 1

        # ── Priority 2: AutoLaneDetector ───────────────────────────
        print("[LaneDetector] LANE_ZONES empty — launching AutoLaneDetector …")
        try:
            from detector.auto_lane_detector import AutoLaneDetector
            ald   = AutoLaneDetector()
            zones = ald.detect(frames, video_path=video_path)

            if zones:
                # Persist zones to config.py for future runs
                ald.save_to_config(zones, video_path)
                self._lane_fracs = zones
                self._free_fracs = {}
                self.calibrated  = True
                n = len(zones)
                print(f"[LaneDetector] ✓ AutoLaneDetector: {n} zones detected: "
                      f"{list(zones.keys())}")
                return n >= 1
            else:
                print("[LaneDetector] ⚠  AutoLaneDetector returned no zones — "
                      "falling back to Hough.")
        except Exception as exc:
            import traceback
            print(f"[LaneDetector] ✗ AutoLaneDetector error: {exc}")
            traceback.print_exc()

        # ── Priority 3: Legacy Hough-based detection ───────────────
        print("[LaneDetector] Running Hough line detection …")
        all_lines: list = []
        for frame in frames:
            lines = self._detect_lines_in_frame(frame)
            if lines:
                all_lines.extend(lines)

        print(f"[LaneDetector] Calibration: {len(all_lines)} raw line segments "
              f"from {len(frames)} frames.")

        if len(all_lines) < 4:
            print("[LaneDetector] Too few lines — switching to fallback mode.")
            self._build_fallback_polygons()
            self.calibrated = True
            return bool(self.lane_polygons)

        boundaries = self._cluster_to_boundaries(
            all_lines, self._orig_w, self._orig_h
        )
        print(f"[LaneDetector] {len(boundaries)} distinct lane boundaries found.")

        if len(boundaries) < 2:
            print("[LaneDetector] Fewer than 2 boundaries — switching to fallback.")
            self._build_fallback_polygons()
            self.calibrated = True
            return bool(self.lane_polygons)

        self._build_lane_polygons(boundaries)
        n = len(self.lane_polygons)
        print(f"[LaneDetector] Calibration done — {n} lane(s) detected: "
              f"{list(self.lane_polygons.keys())}")
        self.calibrated = True
        return n >= 1

    # ----------------------------------------------------------------

    def assign_vehicles_to_lanes(self, detections: list,
                                   frame_w: int = 1280,
                                   frame_h: int = 720) -> dict:
        """
        Assign each detection to a lane polygon based solely on how much
        of the bounding box overlaps the polygon mask.

        A vehicle is assigned to a lane when ≥ OVERLAP_THRESHOLD of its
        bounding-box area falls inside that lane's polygon.  This handles:
          - Cars at the stop line whose centre is just past the boundary
          - Small/distant cars in North/South whose bbox is fully inside
          - Cars partially entering the intersection from the lane

        Vehicles with < OVERLAP_THRESHOLD overlap in every lane are
        silently discarded (no nearest-lane fallback).

        Parameters
        ----------
        detections : list of dicts from YOLODetector.detect()
        frame_w    : width  of the frame detections were made on
        frame_h    : height of the frame detections were made on

        Returns {lane_name: [det, ...]}
        """
        # Minimum fraction of bbox that must overlap the lane polygon.
        # 0.25 = at least a quarter of the car must be in the zone.
        # The overlap test alone is sufficient because:
        #   • A car centred in lane A cannot have 25% overlap with lane B
        #     (lanes are non-overlapping by design).
        #   • Cars just past the stop line still have most of their body inside.
        OVERLAP_THRESHOLD = 0.10

        result = {name: [] for name in self._lane_fracs}

        if not self.calibrated:
            return result

        # Ensure masks are built at the correct frame size
        self._refresh_polys(frame_w, frame_h)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            assigned = False

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
                    break   # assign to first matching lane only

        return result

    # ----------------------------------------------------------------

    def draw_lanes(self, frame: np.ndarray,
                   lane_stats: dict = None) -> np.ndarray:
        """
        Draw clean colored boxes for each lane direction.
        Shows: big direction name (NORTH/SOUTH/EAST/WEST) + car count.
        Polygons are always computed from the actual frame size.
        """
        h, w = frame.shape[:2]

        # Rebuild pixel polygons if frame size changed
        self._refresh_polys(w, h)

        for i, (lane, poly) in enumerate(self.lane_polygons.items()):
            if poly is None or len(poly) == 0:
                continue

            color = LANE_PALETTE[i % len(LANE_PALETTE)]

            # Semi-transparent colored fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

            # Bold colored border
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=4)

            # Centroid for label placement
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # ── Label: lane name + car count + density ────────────────
            if lane_stats and lane in lane_stats:
                count   = lane_stats[lane]["count"]
                density = lane_stats[lane]["density"]
                sub = f"{lane}: {count} cars [{density}]"
            else:
                sub = lane

            (sw, sh), _ = cv2.getTextSize(
                sub, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
            )
            cv2.rectangle(frame,
                          (cx - sw//2 - 6, cy - sh - 6),
                          (cx + sw//2 + 6, cy + 6),
                          (0, 0, 0), -1)
            cv2.putText(frame, sub,
                        (cx - sw//2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        color, 2, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------------

    def get_polygons_serializable(self) -> dict:
        """Return lane polygons as plain Python lists (JSON-serialisable)."""
        return {
            lane: poly.tolist() if poly is not None else []
            for lane, poly in self.lane_polygons.items()
        }

    # ═══════════════════════════════════════════════════════════════
    #  Internal CV Pipeline
    # ═══════════════════════════════════════════════════════════════

    def _detect_lines_in_frame(self, frame: np.ndarray) -> list:
        """
        Full CV pipeline on one frame.
        Returns list of (x1, y1, x2, y2) in ORIGINAL frame coordinates.
        """
        orig_h, orig_w = frame.shape[:2]

        # ── Downscale for fast CV processing ──────────────────────
        if orig_w > _CV_W or orig_h > _CV_H:
            proc = cv2.resize(frame, (_CV_W, _CV_H),
                              interpolation=cv2.INTER_LINEAR)
            sx = orig_w / _CV_W
            sy = orig_h / _CV_H
        else:
            proc = frame
            sx = sy = 1.0

        ph, pw = proc.shape[:2]

        # ── Step 1: Grayscale ──────────────────────────────────────
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Gaussian blur (reduce noise) ───────────────────
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # ── Step 3: Canny edge detection ───────────────────────────
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

        # ── Step 4: Road mask (exclude sky / background) ──────────
        mask = self._build_road_mask(pw, ph)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        # ── Step 5: Dilate — close small gaps in lane markings ─────
        kernel = np.ones((3, 3), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=1)

        # ── Step 6: Hough probabilistic line transform ─────────────
        lines = cv2.HoughLinesP(
            edges,
            rho           = 1,
            theta         = np.pi / 180,
            threshold     = HOUGH_THRESHOLD,
            minLineLength = HOUGH_MIN_LINE_LEN,
            maxLineGap    = HOUGH_MAX_LINE_GAP,
        )

        if lines is None:
            return []

        # ── Step 7: Angle filter ───────────────────────────────────
        filtered = []
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                angle = 90.0
            else:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if LANE_ANGLE_MIN <= angle <= LANE_ANGLE_MAX:
                # Scale back to original frame coordinates
                filtered.append((
                    int(x1 * sx), int(y1 * sy),
                    int(x2 * sx), int(y2 * sy),
                ))

        return filtered

    # ----------------------------------------------------------------

    @staticmethod
    def _build_road_mask(w: int, h: int) -> np.ndarray:
        """Build a binary mask from the ROAD_MASK_POLY config."""
        mask = np.zeros((h, w), dtype=np.uint8)
        pts  = np.array(
            [(int(x * w), int(y * h)) for x, y in ROAD_MASK_POLY],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
        return mask

    # ----------------------------------------------------------------

    def _cluster_to_boundaries(self, lines: list,
                                w: int, h: int) -> list:
        """
        Cluster raw Hough segments into distinct lane-boundary lines.

        Strategy
        --------
        1. Represent each segment by (angle %, x-at-midY, full segment).
        2. Sort by x-at-midY (left → right ordering).
        3. Iteratively merge segments whose midpoints are within
           LINE_CLUSTER_DIST pixels AND whose angles differ by less than
           LINE_CLUSTER_ANGLE_DIFF degrees.
        4. For each cluster, fit a line with cv2.fitLine and extend it
           to span the full road-mask height.

        Returns
        -------
        list of (x_top, y_top, x_bottom, y_bottom) — one per boundary.
        """
        mid_y = h // 2
        road_top    = int(ROAD_MASK_POLY[1][1] * h)
        road_bottom = h - 1

        # ── Annotate each segment ──────────────────────────────────
        annotated = []
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                angle  = 90.0
                x_mid  = float(x1)
            else:
                slope  = (y2 - y1) / (x2 - x1)
                angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                x_mid  = x1 + (mid_y - y1) / slope if slope != 0 else float(x1)
            annotated.append((x_mid, angle, x1, y1, x2, y2))

        # Sort left → right
        annotated.sort(key=lambda a: a[0])

        # ── Iterative cluster merging ──────────────────────────────
        if not annotated:
            return []

        clusters   = [[annotated[0]]]
        for item in annotated[1:]:
            x_mid, angle = item[0], item[1]
            merged = False
            # Try to add to the LAST open cluster
            last_cluster = clusters[-1]
            last_x     = np.mean([a[0] for a in last_cluster])
            last_angle = np.mean([a[1] for a in last_cluster])
            if (abs(x_mid - last_x)   <= LINE_CLUSTER_DIST and
                abs(angle - last_angle) <= LINE_CLUSTER_ANGLE_DIFF):
                last_cluster.append(item)
                merged = True
            if not merged:
                clusters.append([item])

        # ── Fit one line per cluster ───────────────────────────────
        boundaries = []
        for cluster in clusters[: MAX_LANES + 1]:
            pts_x = [a[2] for a in cluster] + [a[4] for a in cluster]
            pts_y = [a[3] for a in cluster] + [a[5] for a in cluster]
            pts   = np.array(list(zip(pts_x, pts_y)), dtype=np.float32)

            if len(pts) < 2:
                continue

            result = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            vx, vy, cx_f, cy_f = (
                float(result[0]), float(result[1]),
                float(result[2]), float(result[3])
            )

            if abs(vy) < 1e-6:
                continue  # near-horizontal fitted line → skip

            # Extend line to road_top and road_bottom
            t_top    = (road_top    - cy_f) / vy
            t_bottom = (road_bottom - cy_f) / vy

            x_top    = int(cx_f + t_top    * vx)
            x_bottom = int(cx_f + t_bottom * vx)

            boundaries.append((x_top, road_top, x_bottom, road_bottom))

        return boundaries

    # ----------------------------------------------------------------

    def _build_lane_polygons(self, boundaries: list):
        """
        Pair adjacent lane boundaries to form quadrilateral lane polygons.
        Polygons are stored in self.lane_polygons as int32 NumPy arrays
        suitable for cv2.pointPolygonTest and cv2.fillPoly.

        Parameters
        ----------
        boundaries : list of (x_top, y_top, x_bottom, y_bottom)
                     sorted left → right (or any consistent direction).
        """
        # Sort left → right by average x position
        boundaries.sort(key=lambda b: (b[0] + b[2]) / 2)

        self.lane_polygons = {}
        for i in range(len(boundaries) - 1):
            left  = boundaries[i]
            right = boundaries[i + 1]
            name  = f"Lane {i + 1}"

            # Quadrilateral: top-left, top-right, bottom-right, bottom-left
            poly = np.array([
                [left[0],  left[1]],    # top of left boundary
                [right[0], right[1]],   # top of right boundary
                [right[2], right[3]],   # bottom of right boundary
                [left[2],  left[3]],    # bottom of left boundary
            ], dtype=np.int32).reshape((-1, 1, 2))

            self.lane_polygons[name] = poly

    # ----------------------------------------------------------------

    def _refresh_polys(self, w: int, h: int):
        """
        (Re)build pixel polygons and pre-computed masks from fractional coords
        whenever the frame size changes.  This ensures correct coordinates
        regardless of the raw video resolution vs. stream resolution mismatch.
        """
        if (w == self._last_w and h == self._last_h
                and self.lane_polygons and self._lane_masks):
            return  # already up to date

        self._last_w, self._last_h = w, h

        self.lane_polygons = {}
        self._lane_masks   = {}
        for name, frac_pts in self._lane_fracs.items():
            pts = np.array(
                [[int(xf * w), int(yf * h)] for xf, yf in frac_pts],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            self.lane_polygons[name] = pts

            # Pre-compute filled mask for fast overlap tests
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            self._lane_masks[name] = mask

        self.free_turn_polygons = {}
        for name, frac_pts in self._free_fracs.items():
            pts = np.array(
                [[int(xf * w), int(yf * h)] for xf, yf in frac_pts],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            self.free_turn_polygons[name] = pts

    def _build_manual_polygons(self):
        """Legacy — kept for Hough path compatibility."""
        w = self._orig_w or 1280
        h = self._orig_h or 720
        self._lane_fracs = dict(LANE_ZONES)
        self._refresh_polys(w, h)
        print(f"[LaneDetector] Signal zones loaded: "
              f"{list(self.lane_polygons.keys())}")

    def _build_free_turn_polygons(self):
        """Legacy — kept for Hough path compatibility."""
        self._free_fracs = dict(FREE_TURN_ZONES) if FREE_TURN_ZONES else {}
        print(f"[LaneDetector] Free-turn zones loaded: "
              f"{list(self._free_fracs.keys())}")

    def _build_fallback_polygons(self):
        """
        Fallback: divide the road-mask bounding box into EXPECTED_LANES
        equal vertical strips. Better than full-frame quadrants because
        it still respects the configured road area.
        """
        w = self._orig_w or 1920
        h = self._orig_h or 1080

        road_top    = int(ROAD_MASK_POLY[1][1] * h)
        road_bottom = h - 1
        road_left   = int(min(p[0] for p in ROAD_MASK_POLY) * w)
        road_right  = int(max(p[0] for p in ROAD_MASK_POLY) * w)
        lane_w      = max(1, (road_right - road_left) // EXPECTED_LANES)

        self.lane_polygons = {}
        for i in range(EXPECTED_LANES):
            x1   = road_left + i       * lane_w
            x2   = road_left + (i + 1) * lane_w
            name = f"Lane {i + 1}"
            poly = np.array([
                [x1, road_top],
                [x2, road_top],
                [x2, road_bottom],
                [x1, road_bottom],
            ], dtype=np.int32).reshape((-1, 1, 2))
            self.lane_polygons[name] = poly

        print(f"[LaneDetector] Fallback: {EXPECTED_LANES} equal strips "
              f"within road mask ({road_left}–{road_right} px wide).")

    # ----------------------------------------------------------------

    def _nearest_lane(self, cx: float, cy: float) -> str | None:
        """Return the lane name whose polygon centroid is nearest to (cx, cy)."""
        min_dist = float("inf")
        nearest  = None
        for lane, poly in self.lane_polygons.items():
            if poly is None or len(poly) == 0:
                continue
            M = cv2.moments(poly)
            if M["m00"] == 0:
                continue
            px   = M["m10"] / M["m00"]
            py   = M["m01"] / M["m00"]
            dist = (px - cx) ** 2 + (py - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest  = lane
        return nearest
