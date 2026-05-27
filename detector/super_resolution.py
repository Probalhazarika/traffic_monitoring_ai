# ─────────────────────────────────────────────────
#  detector/super_resolution.py
#
#  Real-ESRGAN super resolution enhancement.
#
#  Applied only to lane zone crops in Pass 3 of the
#  YOLO detector to upscale tiny distant vehicles
#  before inference.
#
#  The enhancer is lazy-loaded — the heavy model
#  is not downloaded until SR is first enabled via
#  the dashboard toggle.
#
#  Fallback: if Real-ESRGAN is unavailable, uses
#  high-quality bicubic upscaling instead.
# ─────────────────────────────────────────────────

import cv2
import numpy as np
import os, sys


class SuperResolutionEnhancer:
    """
    Wraps Real-ESRGAN for 2× upscaling of lane crops.

    Parameters
    ----------
    device : "mps" | "cuda" | "cpu"
    scale  : upscale factor (2 or 4)
    """

    def __init__(self, device: str = "cpu", scale: int = 2):
        self._device  = device
        self._scale   = scale
        self._upsampler = None
        self._available = False
        self._load_model()

    def _load_model(self):
        """Attempt to load Real-ESRGAN. Falls back to bicubic if unavailable."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=self._scale,
            )

            # Model weights path — auto-downloaded on first use
            model_name = (f"RealESRGAN_x{self._scale}plus.pth"
                          if self._scale > 2
                          else "RealESRGAN_x2plus.pth")
            model_path = os.path.join(
                os.path.expanduser("~"), ".cache", "realesrgan", model_name
            )

            # Map MPS → cpu for realesrgan (MPS not always supported by basicsr)
            sr_device = "cuda" if self._device == "cuda" else "cpu"

            self._upsampler = RealESRGANer(
                scale      = self._scale,
                model_path = model_path,
                model      = model,
                tile       = 0,          # 0 = no tiling (crops are small)
                tile_pad   = 10,
                pre_pad    = 0,
                half       = (self._device == "cuda"),
                device     = sr_device,
            )
            self._available = True
            print(f"[SuperResolution] Real-ESRGAN x{self._scale} loaded ✅")

        except Exception as e:
            print(f"[SuperResolution] Real-ESRGAN unavailable: {e}")
            print("[SuperResolution] Falling back to INTER_CUBIC upscaling")
            self._available = False

    def enhance(self, crop: np.ndarray) -> np.ndarray:
        """
        Upscale a BGR crop by self._scale.

        Parameters
        ----------
        crop : BGR np.ndarray (the lane zone crop)

        Returns
        -------
        Upscaled BGR np.ndarray
        """
        if not self._available or self._upsampler is None:
            # High-quality bicubic fallback
            h, w = crop.shape[:2]
            return cv2.resize(crop, (w * self._scale, h * self._scale),
                              interpolation=cv2.INTER_CUBIC)

        try:
            # Real-ESRGAN expects RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            enhanced_rgb, _ = self._upsampler.enhance(rgb, outscale=self._scale)
            return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[SuperResolution] enhance() error: {e}")
            h, w = crop.shape[:2]
            return cv2.resize(crop, (w * self._scale, h * self._scale),
                              interpolation=cv2.INTER_CUBIC)

    @property
    def available(self) -> bool:
        return self._available
