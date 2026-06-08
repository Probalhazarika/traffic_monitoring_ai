# ─────────────────────────────────────────────────────────────────
#  detector/auto_lane_detector.py
#
#  AutoLaneDetector — fully automatic queue-zone detection
#  for aerial traffic cameras.
#
#  Pipeline
#  ────────
#  Stage 1  Median-blend N frames → stable background (cars removed)
#  Stage 2  Road surface detection via CLAHE + Otsu + morphology
#  Stage 3  Find intersection box via distance transform
#           (thickest / widest road region = where all arms meet)
#  Stage 4  Split road mask into directional arms by angle from
#           intersection centre (atan2 → N/S/E/W sectors)
#  Stage 5  For each arm: trace the FULL VISIBLE road extent to build
#           a tight 4-corner queue-zone polygon — every arm is
#           measured independently so a tiny South arm gets a
#           tiny zone and a long North arm gets a long zone.
#  Stage 6  Save detected zones to config.py keyed to the video
#           filename — future runs skip re-detection for same footage.
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import re


class AutoLaneDetector:
    """
    Automatically detect traffic signal queue zones from aerial video frames.

    Returned zones use fractional (x_frac, y_frac) coordinates — identical
    to the LANE_ZONES format in config.py — so they slot straight into the
    existing LaneDetector pipeline without any changes to the rest of the code.

    Usage
    -----
        ald   = AutoLaneDetector()
        zones = ald.detect(frames, video_path="videos/traffic.mp4")
        ald.save_to_config(zones, video_path="videos/traffic.mp4")
        # zones == {"North": [(xf,yf),...], "South": [...], ...}
    """

    def __init__(self, config_path: str = None):
        # Path to config.py — used when saving detected zones
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.py",
        )

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def detect(self, frames: list, video_path: str = "") -> dict:
        """
        Run the full 5-stage auto-detection pipeline.

        Parameters
        ----------
        frames     : list of BGR numpy arrays (first N calibration frames)
        video_path : path to the video file (for logging / cache key)

        Returns
        -------
        dict  {arm_name: [(xf, yf), ...]}  in LANE_ZONES fractional format.
              Empty dict if detection fails.
        """
        if not frames:
            return {}

        print("[AutoLane] ── Starting automatic lane detection ─────────")

        # ── Stage 1: Stable road background ──────────────────────
        bg    = self._build_background(frames)
        h, w  = bg.shape[:2]
        print(f"[AutoLane] Reference frame: {w}×{h}")

        # ── Stage 2: Road surface mask ────────────────────────────
        road     = self._detect_road(bg, w, h)
        road_px  = int(np.sum(road > 0))
        pct      = 100.0 * road_px / (w * h)
        print(f"[AutoLane] Road pixels detected: {road_px}  ({pct:.1f}% of frame)")
        if road_px < w * h * 0.03:
            print("[AutoLane] ⚠  Very little road visible — detection may be noisy.")

        # ── Stage 3: Intersection centre + bounding box ───────────
        int_box, center = self._find_intersection(road, w, h)
        if center is None:
            print("[AutoLane] ✗  Could not locate intersection centre — aborting.")
            return {}
        cx, cy = center
        print(f"[AutoLane] Intersection centre: ({cx}, {cy})   box: {int_box}")

        # ── Stage 4: Directional arm splitting ────────────────────
        arms = self._split_arms(road, cx, cy, int_box, w, h)
        print(f"[AutoLane] Arms found: {list(arms.keys())}")
        if not arms:
            print("[AutoLane] ✗  No road arms detected — aborting.")
            return {}

        # ── Stage 5: Per-arm zone polygons ────────────────────────
        zones = self._arms_to_zones(arms, cx, cy, int_box, w, h)

        print("[AutoLane] Detected zones (fractional coords):")
        for name, pts in zones.items():
            print(f"  {name}: {pts}")

        # ── Debug visualisation ───────────────────────────────────
        self._save_debug(bg, road, zones, cx, cy, int_box, w, h)

        print("[AutoLane] ── Detection complete ───────────────────────")
        return zones

    # ----------------------------------------------------------------

    def save_to_config(self, zones: dict, video_path: str = ""):
        """
        Patch the LANE_ZONES block in config.py with the detected zones
        and record the video filename so future runs can skip re-detection.

        Uses a unique sentinel comment '# <<AUTO_LANE_START>>' to reliably
        find and replace the block — prevents duplicates on repeated runs.
        """
        if not zones:
            return

        try:
            with open(self.config_path, "r") as fh:
                content = fh.read()

            video_basename = os.path.basename(video_path) if video_path else "unknown"

            # ── Build the replacement block with a unique sentinel ──
            sentinel_start = "# <<AUTO_LANE_START>>"
            sentinel_end   = "# <<AUTO_LANE_END>>"
            lines = [
                sentinel_start,
                f"# AUTO-DETECTED for video: {video_basename}",
                f'AUTO_LANE_VIDEO = "{video_basename}"',
                "LANE_ZONES = {",
            ]
            for name, pts in zones.items():
                pts_str = ", ".join(f"({x}, {y})" for x, y in pts)
                lines.append(f'    "{name}": [{pts_str}],')
            lines.append("}")
            lines.append(sentinel_end)
            new_block = "\n".join(lines)

            # ── Strategy 1: Replace between existing sentinels ───────
            sentinel_pattern = re.escape(sentinel_start) + r".*?" + re.escape(sentinel_end)
            replaced, n_subs = re.subn(sentinel_pattern, new_block,
                                        content, flags=re.DOTALL)

            if n_subs > 0:
                with open(self.config_path, "w") as fh:
                    fh.write(replaced)
                print(f"[AutoLane] ✓ Zones updated in config.py  (video: {video_basename})")
                return

            # ── Strategy 2: Replace the LANE_ZONES = { ... } block ───
            # Matches: optional prior AUTO_LANE_VIDEO line + LANE_ZONES dict
            pattern2 = (
                r"(?:# AUTO-DETECTED[^\n]*\n)?"
                r"(?:AUTO_LANE_VIDEO\s*=\s*[^\n]+\n)?"
                r"LANE_ZONES\s*=\s*\{[^}]*\}"
            )
            replaced, n_subs = re.subn(pattern2, new_block,
                                        content, count=1, flags=re.DOTALL)

            if n_subs > 0:
                with open(self.config_path, "w") as fh:
                    fh.write(replaced)
                print(f"[AutoLane] ✓ Zones saved to config.py  (video: {video_basename})")
                return

            # ── Strategy 3: Append at end as safe fallback ───────────
            with open(self.config_path, "w") as fh:
                fh.write(content.rstrip() + "\n\n" + new_block + "\n")
            print(f"[AutoLane] ✓ Zones appended to config.py  (video: {video_basename})")

        except Exception as exc:
            print(f"[AutoLane] ✗ Could not save to config.py: {exc}")

    # ═══════════════════════════════════════════════════════════════
    #  Stage 1 — Stable Background
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_background(frames: list) -> np.ndarray:
        """
        Median-blend up to 60 frames to eliminate moving vehicles.

        Why median?  Moving vehicles appear at different pixel positions
        each frame — the median value across all frames is the static
        road surface that lies beneath them.
        """
        sample = frames[::2][:60]           # every other frame, cap at 60
        stack  = np.stack([f.astype(np.uint8) for f in sample], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2 — Road Surface Detection
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_road(frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Binary road mask: bright-ish, medium-saturation asphalt.
        Uses CLAHE and a fixed lower threshold to capture roads even in dark night scenes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Local illumination normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        eq = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(eq, (21, 21), 0)
        
        # Lower threshold: anything above 40 in CLAHE image is probably road/ground
        _, th1 = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        
        # Exclude extreme glare (headlights, lamps)
        _, no_hi = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        
        road = cv2.bitwise_and(th1, no_hi)
        
        # Close small gaps (lane markings, crosswalks interrupting the surface)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k_close)
        
        # Keep only the largest connected component (the main intersection)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road, 8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean = (labels == largest_label).astype(np.uint8) * 255
        else:
            clean = road
            
        return clean

    # ═══════════════════════════════════════════════════════════════
    #  Stage 3 — Intersection Centre
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _find_intersection(road: np.ndarray, w: int, h: int):
        """
        Locate the intersection box via the distance transform.

        The distance transform assigns each road pixel a value equal to
        its distance from the nearest road edge.  Where multiple arms meet
        (the intersection) the road is widest → those pixels have the
        highest distance values.  Thresholding at ~50 % isolates this region.

        Returns
        -------
        (int_box, (cx, cy))  where int_box = (x1, y1, x2, y2)
        """
        dist   = cv2.distanceTransform(road, cv2.DIST_L2, 5)
        d_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thick = cv2.threshold(d_norm, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: centre-of-mass of the whole road mask
            M = cv2.moments(road)
            if M["m00"] == 0:
                cx, cy = w // 2, h // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            pad = min(w, h) // 10
            return (cx - pad, cy - pad, cx + pad, cy + pad), (cx, cy)

        # Score each candidate on:
        #   1. Area (bigger = more likely intersection)
        #   2. Closeness to frame centre
        #   3. Roundness / compactness (intersection box tends to be squarish)
        fc_x, fc_y       = w / 2.0, h / 2.0
        best_cnt, best_s = None, -1e18

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            mx    = M["m10"] / M["m00"]
            my    = M["m01"] / M["m00"]
            d     = np.hypot(mx - fc_x, my - fc_y)
            # Compactness: 1.0 = perfect circle, lower = elongated arm
            perimeter   = cv2.arcLength(cnt, True)
            compactness = (4 * np.pi * area / (perimeter ** 2 + 1e-6))
            # Score: reward area + compactness, penalise distance from frame centre
            score = area * (1 + compactness) - d * 1.5
            if score > best_s:
                best_s   = score
                best_cnt = cnt

        if best_cnt is None:
            return (w//4, h//4, 3*w//4, 3*h//4), (w//2, h//2)

        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        M  = cv2.moments(best_cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Constrain the intersection box to prevent it from swallowing the arms
        # Max width and height = 25% of frame
        bw = min(bw, int(w * 0.25))
        bh = min(bh, int(h * 0.25))
        
        # Re-center the constrained box
        bx = max(0, cx - bw // 2)
        by = max(0, cy - bh // 2)

        return (bx, by, bx + bw, by + bh), (cx, cy)

    # ═══════════════════════════════════════════════════════════════
    #  Stage 4 — Arm Splitting (vectorised)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _split_arms(road: np.ndarray, cx: int, cy: int,
                    int_box, w: int, h: int) -> dict:
        """
        Partition road pixels into North / South / East / West arm masks
        using the angle each pixel makes relative to the intersection centre.

        Angle convention  (numpy atan2, image coords where Y↓):
          0°   → East   (right)
          90°  → South  (downward in image)
         -90°  → North  (upward in image)
         ±180° → West   (left)

        Pixels inside the intersection box itself are excluded — arm zones
        start at the stop line (the edge of the intersection box).
        """
        ys, xs = np.where(road > 0)
        if len(xs) == 0:
            return {}

        # Vectorised angle computation — no Python loops over pixels
        dx     = xs.astype(np.float32) - cx
        dy     = ys.astype(np.float32) - cy
        angles = np.degrees(np.arctan2(dy, dx))    # range: -180 to 180

        # Exclude pixels too close to the intersection box centroid
        # (the threshold is 45 % of the intersection box diagonal)
        ix1, iy1, ix2, iy2 = int_box
        box_diag = np.hypot(ix2 - ix1, iy2 - iy1) * 0.45
        beyond   = np.hypot(dx, dy) > max(20, box_diag)

        # 45°-wide sector definitions
        sector_defs = {
            "North": (angles > -135) & (angles < -45)   & beyond,
            "East":  (angles >  -45) & (angles <  45)   & beyond,
            "South": (angles >   45) & (angles < 135)   & beyond,
            "West":  ((angles > 135) | (angles < -135)) & beyond,
        }

        # Minimum pixels for an arm to be considered real
        min_px = max(400, w * h * 0.002)    # ≥ 0.2 % of frame
        arms   = {}

        for direction, sel in sector_defs.items():
            n_px = int(np.sum(sel))
            if n_px >= min_px:
                mask = np.zeros_like(road)
                mask[ys[sel], xs[sel]] = 255
                arms[direction] = mask
                print(f"[AutoLane] {direction:5s}: {n_px} road pixels")
            else:
                print(f"[AutoLane] {direction:5s}: only {n_px} px — skipped "
                      f"(min {int(min_px)})")

        return arms

    # ═══════════════════════════════════════════════════════════════
    #  Stage 5 — Zone Polygons (per-arm adaptive extent)
    # ═══════════════════════════════════════════════════════════════

    def _arms_to_zones(self, arms: dict, cx: int, cy: int,
                       int_box, w: int, h: int) -> dict:
        """
        Convert each arm mask into a 4-corner queue-zone polygon.

        Key design decision (per user requirement):
          The zone always covers the FULL VISIBLE road extent of that arm —
          from the stop line (intersection box edge) to wherever the road
          disappears from the frame.  Each arm is measured independently,
          so a short South arm gets a tiny zone and a long North arm gets
          a long zone — just like the manual zones.
        """
        ix1, iy1, ix2, iy2 = int_box
        zones = {}

        for direction, arm_mask in arms.items():
            ys_arm, xs_arm = np.where(arm_mask > 0)
            if len(xs_arm) == 0:
                continue

            try:
                if direction == "North":
                    # Stop line = top edge of intersection box
                    # Visible end = topmost road pixel (min Y)
                    corners = self._vertical_zone(
                        xs_arm, ys_arm,
                        y_stop=iy1,
                        y_far=int(np.min(ys_arm)),
                        direction=direction
                    )

                elif direction == "South":
                    # Stop line = bottom edge of intersection box
                    # Visible end = bottommost road pixel (max Y)
                    corners = self._vertical_zone(
                        xs_arm, ys_arm,
                        y_stop=iy2,
                        y_far=int(np.max(ys_arm)),
                        direction=direction
                    )

                elif direction == "East":
                    # Stop line = right edge of intersection box
                    # Visible end = rightmost road pixel (max X)
                    corners = self._horizontal_zone(
                        xs_arm, ys_arm,
                        x_stop=ix2,
                        x_far=int(np.max(xs_arm)),
                        direction=direction
                    )

                else:  # West
                    # Stop line = left edge of intersection box
                    # Visible end = leftmost road pixel (min X)
                    corners = self._horizontal_zone(
                        xs_arm, ys_arm,
                        x_stop=ix1,
                        x_far=int(np.min(xs_arm)),
                        direction=direction
                    )

                if corners and len(corners) == 4:
                    frac = [
                        (round(max(0.0, min(1.0, px / w)), 4),
                         round(max(0.0, min(1.0, py / h)), 4))
                        for px, py in corners
                    ]
                    zones[direction] = frac

            except Exception as exc:
                print(f"[AutoLane] Zone computation error ({direction}): {exc}")

        return zones

    # ── Polygon shape helpers ────────────────────────────────────

    @staticmethod
    def _vertical_zone(xs: np.ndarray, ys: np.ndarray,
                        y_stop: int, y_far: int, direction: str) -> list:
        """
        Build a 4-corner polygon for a vertical (North / South) arm.

        Scans 30 horizontal Y-slices across the full arm extent to
        measure the actual road width at each level.  Uses the 15th and
        85th percentile of observed edges to ignore stray pixels at the
        road margins (parked cars, shadows, kerbs, etc.).

        Returns [(tl), (tr), (br), (bl)] as pixel coordinates.
        """
        y_lo, y_hi = min(y_stop, y_far), max(y_stop, y_far)
        if y_hi <= y_lo:
            return []

        step    = max(1, (y_hi - y_lo) // 30)
        x_lefts, x_rights = [], []

        for y in range(y_lo, y_hi + 1, step):
            # Pixels in a band ±step around this Y level
            band = xs[(ys >= max(0, y - step)) & (ys <= y + step)]
            if len(band) == 0:
                continue
            x_lefts.append(int(np.min(band)))
            x_rights.append(int(np.max(band)))

        if not x_lefts:
            return []

        x_left  = int(np.percentile(x_lefts,  15))
        x_right = int(np.percentile(x_rights, 85))
        
        mid_x = (x_left + x_right) // 2
        
        if direction == "North":
            # North inbound is left half
            return [
                (x_left, y_lo),
                (mid_x, y_lo),
                (mid_x, y_hi),
                (x_left, y_hi),
            ]
        else:
            # South inbound is right half
            return [
                (mid_x, y_lo),
                (x_right, y_lo),
                (x_right, y_hi),
                (mid_x, y_hi),
            ]

    @staticmethod
    def _horizontal_zone(xs: np.ndarray, ys: np.ndarray,
                          x_stop: int, x_far: int, direction: str) -> list:
        """
        Build a 4-corner polygon for a horizontal (East / West) arm.

        Scans 30 vertical X-slices to measure the actual road height
        at each column position.
        """
        x_lo, x_hi = min(x_stop, x_far), max(x_stop, x_far)
        if x_hi <= x_lo:
            return []

        step    = max(1, (x_hi - x_lo) // 30)
        y_tops, y_bots = [], []

        for x in range(x_lo, x_hi + 1, step):
            band = ys[(xs >= max(0, x - step)) & (xs <= x + step)]
            if len(band) == 0:
                continue
            y_tops.append(int(np.min(band)))
            y_bots.append(int(np.max(band)))

        if not y_tops:
            return []

        y_top = int(np.percentile(y_tops, 15))
        y_bot = int(np.percentile(y_bots, 85))
        
        mid_y = (y_top + y_bot) // 2
        
        if direction == "East":
            # East inbound is top half
            return [
                (x_lo, y_top),
                (x_hi, y_top),
                (x_hi, mid_y),
                (x_lo, mid_y),
            ]
        else:
            # West inbound is bottom half
            return [
                (x_lo, mid_y),
                (x_hi, mid_y),
                (x_hi, y_bot),
                (x_lo, y_bot),
            ]

    # ═══════════════════════════════════════════════════════════════
    #  Debug Visualisation
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _save_debug(bg: np.ndarray, road: np.ndarray,
                    zones: dict, cx: int, cy: int,
                    int_box, w: int, h: int):
        """
        Save an annotated JPEG (auto_lane_debug.jpg) in the project root
        so the user can visually verify the detected zones.

        Shows:
          • Green overlay  — detected road surface mask
          • Cyan rectangle — detected intersection box
          • Red dot        — intersection centre
          • Coloured polygons — detected queue zones with labels
        """
        try:
            vis = bg.copy()

            # Road mask as a semi-transparent green tint
            road_color = np.zeros_like(vis)
            road_color[road > 0] = (0, 200, 0)
            vis = cv2.addWeighted(vis, 0.65, road_color, 0.35, 0)

            # Intersection box
            ix1, iy1, ix2, iy2 = int_box
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(vis, "Intersection", (ix1, max(0, iy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # Zone polygons with semi-transparent fill
            palette = [(0,255,0),(255,165,0),(0,165,255),(255,0,255),
                       (0,255,255),(255,255,0)]
            for i, (name, frac_pts) in enumerate(zones.items()):
                color  = palette[i % len(palette)]
                pts_px = np.array(
                    [[int(x * w), int(y * h)] for x, y in frac_pts],
                    dtype=np.int32,
                )
                # Transparent fill
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts_px], color)
                vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0)
                # Bold border
                cv2.polylines(vis, [pts_px], True, color, 3)
                # Label at polygon centroid
                M = cv2.moments(pts_px)
                if M["m00"]:
                    lx = int(M["m10"] / M["m00"])
                    ly = int(M["m01"] / M["m00"])
                    cv2.putText(vis, name, (lx - 30, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                                color, 2, cv2.LINE_AA)

            # Save next to config.py
            out_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            out_path = os.path.join(out_dir, "auto_lane_debug.jpg")
            cv2.imwrite(out_path, vis)
            print(f"[AutoLane] Debug image saved → {out_path}")

        except Exception as exc:
            print(f"[AutoLane] Could not save debug image: {exc}")
