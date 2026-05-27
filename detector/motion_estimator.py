# ─────────────────────────────────────────────────
#  detector/motion_estimator.py
#
#  Computes per-lane motion density using sparse
#  Lucas-Kanade optical flow between consecutive frames.
#
#  Motion density = mean magnitude of feature-point
#  displacements inside each lane polygon, normalised
#  to [0, 1].  Even when YOLO misses distant vehicles,
#  optical flow can still detect that pixels are moving.
# ─────────────────────────────────────────────────

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LANE_ZONES

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize   = (15, 15),
    maxLevel  = 2,
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Shi-Tomasi corner detection parameters (feature points to track)
FEATURE_PARAMS = dict(
    maxCorners    = 200,
    qualityLevel  = 0.3,
    minDistance   = 7,
    blockSize     = 7,
)

# Work at this scale to keep optical flow fast
FLOW_SCALE = 0.5

# Magnitude above which motion is considered "significant" (normalisation ceiling)
MAX_MAGNITUDE = 10.0


class MotionEstimator:
    """
    Estimates per-lane motion density between consecutive frames.

    Usage
    -----
    estimator = MotionEstimator()
    ...
    scores = estimator.update(gray_frame, frame_w, frame_h)
    # scores → {lane_name: float in [0, 1]}
    """

    def __init__(self):
        self._prev_gray  = None   # grayscale from previous frame (at FLOW_SCALE)
        self._prev_pts   = None   # tracked feature points (at FLOW_SCALE)
        self._lane_masks = {}     # cached binary masks per lane
        self._mask_size  = None   # (w, h) the masks were built for

    # ── Public API ──────────────────────────────────────────

    def update(self, gray_frame: np.ndarray,
               frame_w: int, frame_h: int) -> dict:
        """
        Compute per-lane motion density.

        Parameters
        ----------
        gray_frame : grayscale uint8 frame at display resolution
        frame_w, frame_h : display frame dimensions

        Returns
        -------
        {lane_name: float}  — motion density in [0, 1] per lane
        """
        # Downscale for speed
        sw = int(frame_w  * FLOW_SCALE)
        sh = int(frame_h  * FLOW_SCALE)
        small = cv2.resize(gray_frame, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # Build lane masks if frame size changed
        if self._mask_size != (sw, sh):
            self._build_masks(sw, sh)
            self._mask_size = (sw, sh)

        # Default: zero motion
        scores = {lane: 0.0 for lane in LANE_ZONES}

        if self._prev_gray is None:
            self._prev_gray = small
            self._prev_pts  = self._detect_features(small)
            return scores

        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_gray = small
            self._prev_pts  = self._detect_features(small)
            return scores

        # Compute Lucas-Kanade optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, small, self._prev_pts, None, **LK_PARAMS
        )

        if new_pts is None:
            self._prev_gray = small
            self._prev_pts  = self._detect_features(small)
            return scores

        # Keep only successfully tracked points
        good_old = self._prev_pts[status.flatten() == 1]
        good_new = new_pts[status.flatten() == 1]

        if len(good_old) == 0:
            self._prev_gray = small
            self._prev_pts  = self._detect_features(small)
            return scores

        # Compute displacement magnitude for each tracked point
        delta    = good_new - good_old
        magnitudes = np.linalg.norm(delta, axis=1)

        # Per-lane: collect magnitudes of points inside lane mask
        for lane, mask in self._lane_masks.items():
            lane_mags = []
            for pt, mag in zip(good_old, magnitudes):
                px = int(pt[0, 0])
                py = int(pt[0, 1])
                if 0 <= px < sw and 0 <= py < sh and mask[py, px]:
                    lane_mags.append(mag)

            if lane_mags:
                mean_mag = np.mean(lane_mags)
                scores[lane] = float(min(1.0, mean_mag / MAX_MAGNITUDE))

        # Refresh: re-detect features every ~30 calls to avoid drift
        self._prev_gray = small
        if len(good_new) < 30:
            self._prev_pts = self._detect_features(small)
        else:
            self._prev_pts = good_new.reshape(-1, 1, 2)

        return scores

    # ── Internal helpers ────────────────────────────────────

    def _detect_features(self, gray_small: np.ndarray) -> np.ndarray | None:
        """Run Shi-Tomasi on the downscaled grayscale frame."""
        pts = cv2.goodFeaturesToTrack(gray_small, mask=None, **FEATURE_PARAMS)
        return pts

    def _build_masks(self, sw: int, sh: int) -> None:
        """Build binary masks for each lane zone at FLOW_SCALE resolution."""
        self._lane_masks = {}
        for lane, frac_pts in LANE_ZONES.items():
            mask = np.zeros((sh, sw), dtype=np.uint8)
            px_pts = np.array(
                [(int(x * sw), int(y * sh)) for x, y in frac_pts],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [px_pts], 255)
            self._lane_masks[lane] = mask.astype(bool)
