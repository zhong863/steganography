"""Optional blind receiver denoising; no attack label or clean image needed."""

import numpy as np


def estimate_noise_sigma(image_unit):
    image = np.asarray(image_unit, dtype=np.float32)
    # Unit-energy Haar diagonal detail, evaluated on non-overlapping 2x2 blocks.
    detail = (image[0::2, 0::2] - image[0::2, 1::2]
              - image[1::2, 0::2] + image[1::2, 1::2]) / 2
    return float(np.median(np.abs(detail - np.median(detail))) / .67448975)


def denoise_received(image_unit):
    import cv2
    from .attacks import to_uint8

    sigma = estimate_noise_sigma(image_unit) * 255
    if sigma <= 5:
        return image_unit
    # Quantization occurs inside the receiver, after the prescribed float attack.
    denoised = cv2.fastNlMeansDenoising(to_uint8(image_unit), None,
                                      h=float(.8 * sigma), templateWindowSize=7,
                                      searchWindowSize=21)
    return denoised.astype(np.float32) / 255
