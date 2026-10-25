"""Explicit, independently applied JPEG / AWGN / median protocols."""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

ATTACKS = ("jpeg", "awgn", "mblur")


@dataclass(frozen=True)
class AttackConfig:
    jpeg_quality: int = 70
    jpeg_subsampling: int = 2
    noise_sigma: float = 0.05
    noise_domain: str = "unit"
    median_kernel: int = 5

    def __post_init__(self):
        if not 1 <= self.jpeg_quality <= 100 or self.jpeg_subsampling not in (0, 1, 2):
            raise ValueError("Invalid JPEG quality or subsampling")
        if not np.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and nonnegative")
        if self.noise_domain not in ("unit", "signed"):
            raise ValueError("noise_domain must be unit ([0,1]) or signed ([-1,1])")
        if self.median_kernel < 3 or self.median_kernel % 2 != 1:
            raise ValueError("median_kernel must be odd and at least 3")


def to_uint8(unit_rgb):
    # Match torchvision ToPILImage's truncation instead of changing rounding.
    return (np.clip(unit_rgb, 0, 1) * 255).astype(np.uint8)


def apply_attack(clean_rgb, attack, config=AttackConfig(), seed=0):
    """Input is the saved clean uint8 RGB image; output is float32 [0,1].

    AWGN stays floating point, matching the baseline's in-memory transmission.
    The runner saves this exact float array as .npy alongside its PNG preview.
    """
    clean_rgb = np.asarray(clean_rgb)
    if clean_rgb.dtype != np.uint8 or clean_rgb.ndim != 3 or clean_rgb.shape[-1] != 3:
        raise ValueError("Expected an HWC uint8 RGB image")
    if attack == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(clean_rgb).save(buffer, format="JPEG", quality=config.jpeg_quality,
                                       subsampling=config.jpeg_subsampling)
        buffer.seek(0)
        with Image.open(buffer) as image:
            result = np.array(image.convert("RGB"))
        return result.astype(np.float32) / 255
    if attack == "mblur":
        import cv2
        return cv2.medianBlur(clean_rgb, config.median_kernel).astype(np.float32) / 255
    if attack == "awgn":
        sigma = config.noise_sigma / (2 if config.noise_domain == "signed" else 1)
        noise = np.random.RandomState(seed).normal(0, sigma, clean_rgb.shape)
        return np.clip(clean_rgb.astype(np.float64) / 255 + noise, 0, 1).astype(np.float32)
    raise ValueError("Unknown attack: %s" % attack)
