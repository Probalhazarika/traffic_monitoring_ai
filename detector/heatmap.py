# ─────────────────────────────────────────────────
#  detector/heatmap.py
#
#  Real-time traffic congestion heatmap.
#
#  Maintains a floating-point accumulator the same
#  size as the display frame.  Every detection pass
#  adds a Gaussian blob at each vehicle bbox centre.
#  The accumulator decays each frame so old data
#  fades out naturally.
#
#  Usage:
#    heatmap = TrafficHeatmap(w=1280, h=720)
#    heatmap.update(detections)
#    frame   = heatmap.overlay(frame, alpha=0.45)
# ─────────────────────────────────────────────────

import cv2
import numpy as np

# Exponential decay factor applied every frame (0 = instant decay, 1 = no decay)
DECAY = 0.94

# Gaussian blob radius (pixels at 1280×720) — larger = smoother heatmap
BLOB_RADIUS = 40

# Minimum accumulator value before normalisation (prevents all-zero map)
EPS = 1e-6


class TrafficHeatmap:
    """
    Maintains a decaying heatmap of vehicle positions.

    Parameters
    ----------
    w, h : display frame dimensions (default 1280×720)
    """

    def __init__(self, w: int = 1280, h: int = 720):
        self.w   = w
        self.h   = h
        self._acc = np.zeros((h, w), dtype=np.float32)

        # Pre-build a Gaussian kernel for fast blob stamping
        self._kernel = self._make_gaussian(BLOB_RADIUS)

    # ── Public API ──────────────────────────────────────────

    def update(self, detections: list) -> None:
        """
        Add vehicle positions to the accumulator and apply decay.

        Parameters
        ----------
        detections : list of detection dicts with 'cx', 'cy' keys
        """
        # Decay first so new detections aren't immediately faded
        self._acc *= DECAY

        kh, kw = self._kernel.shape
        kr = kh // 2   # kernel radius

        for d in detections:
            cx, cy = d["cx"], d["cy"]
            # Compute overlapping region between kernel and accumulator
            x1a = max(0, cx - kr);  x2a = min(self.w, cx + kr + 1)
            y1a = max(0, cy - kr);  y2a = min(self.h, cy + kr + 1)
            x1k = x1a - (cx - kr);  x2k = x1k + (x2a - x1a)
            y1k = y1a - (cy - kr);  y2k = y1k + (y2a - y1a)
            if x2a > x1a and y2a > y1a:
                self._acc[y1a:y2a, x1a:x2a] += self._kernel[y1k:y2k, x1k:x2k]

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Alpha-blend the heatmap over `frame`.

        Parameters
        ----------
        frame : BGR display frame
        alpha : blending weight for heatmap layer

        Returns
        -------
        Annotated frame (in-place modification)
        """
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            # Resize accumulator if display size changed
            self._acc = cv2.resize(self._acc, (frame.shape[1], frame.shape[0]))
            self.w, self.h = frame.shape[1], frame.shape[0]

        # Normalise to [0, 255]
        acc_max = self._acc.max()
        if acc_max < EPS:
            return frame   # nothing to draw

        norm = np.clip(self._acc / acc_max * 255, 0, 255).astype(np.uint8)

        # Apply colormap (JET: blue=cool, red=hot)
        heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Mask out near-zero pixels so background stays clean
        mask = norm > 15
        mask_3c = np.stack([mask, mask, mask], axis=-1)

        # Alpha blend only where heatmap is non-trivial
        blended = frame.copy()
        blended[mask_3c] = cv2.addWeighted(
            frame, 1 - alpha,
            heatmap_bgr, alpha, 0
        )[mask_3c]

        return blended

    def reset(self) -> None:
        """Clear the accumulator (e.g. when video loops)."""
        self._acc[:] = 0.0

    def resize(self, w: int, h: int) -> None:
        """Resize accumulator to new display dimensions."""
        self._acc = cv2.resize(self._acc, (w, h))
        self.w, self.h = w, h

    # ── Internal helpers ────────────────────────────────────

    @staticmethod
    def _make_gaussian(radius: int) -> np.ndarray:
        """Return a normalised 2D Gaussian kernel of given radius."""
        size = 2 * radius + 1
        x = np.linspace(-radius, radius, size)
        gauss_1d = np.exp(-0.5 * (x / (radius / 2.5)) ** 2)
        gauss_2d = np.outer(gauss_1d, gauss_1d)
        return (gauss_2d / gauss_2d.max()).astype(np.float32)
