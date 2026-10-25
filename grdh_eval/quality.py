"""BasicSR NIQE on clean RGB, with explicit color conversion and no crop."""

from pathlib import Path

import numpy as np


def niqe_rgb(image, convert_to="y"):
    from .vendor.basicsr_niqe import niqe

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("NIQE input must be uint8 RGB")
    if min(image.shape[:2]) < 192:
        raise ValueError("NIQE requires images of at least 192 pixels per side")
    if convert_to == "y":
        # MATLAB/BasicSR BT.601 limited-range luma. Input here is RGB, not BGR.
        gray = image.astype(np.float32).dot(np.array([65.481, 128.553, 24.966], dtype=np.float32)) / 255 + 16
    elif convert_to == "gray":
        import cv2
        gray = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY) * 255
    else:
        raise ValueError("NIQE conversion must be y or gray")
    with np.load(Path(__file__).parent / "vendor/niqe_pris_params.npz") as params:
        result = niqe(gray.round(), params["mu_pris_param"], params["cov_pris_param"], params["gaussian_window"])
    if not np.isfinite(result):
        raise ValueError("NIQE is undefined for this image; no score is assigned")
    return float(result)
