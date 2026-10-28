"""Receiver-only 0/1 candidate regeneration for uncertain uncoded bits.

No original image, true message, attack label or sender prompt is accepted.
Candidates for different bits use the same initial estimate (one independent
pass, not an order-dependent greedy search). Nothing is written to disk.
"""

import numpy as np

METRICS = ("mse", "l1", "blur_mse")
ANCHORS = ("continuous", "hard")


def symbols_to_latent(codec, symbols):
    symbols = np.asarray(symbols, dtype=np.float64)
    if symbols.shape != codec.config.shape or not np.isfinite(symbols).all():
        raise ValueError("Invalid candidate symbols")
    standardized = (symbols - codec.mean) / codec.std
    latent = codec.kernel @ standardized @ codec.kernel.T
    return latent.reshape(-1)[codec.permutation].reshape(codec.config.shape).astype(np.float32)


def candidate_pair(codec, values, index, anchor):
    """Change precisely one symbol to 0/1; keep all other symbols identical."""
    if anchor not in ANCHORS or not 0 <= index < values.size:
        raise ValueError("Invalid candidate anchor or position")
    base = values.copy() if anchor == "continuous" else (values > .5).astype(np.float64)
    zero, one = base.copy(), base.copy()
    zero.flat[index], one.flat[index] = 0, 1
    return np.stack([symbols_to_latent(codec, zero), symbols_to_latent(codec, one)])


def _unit(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    else:
        image = image.astype(np.float32)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected finite unit-range images")
    return image


def candidate_distances(images, received):
    import cv2

    images, received = _unit(images), _unit(received)
    if received.ndim != 3 or received.shape[-1] != 3 or images.shape != (2,) + received.shape:
        raise ValueError("Expected exactly two candidate RGB images")
    difference = images.astype(np.float64) - received.astype(np.float64)
    target_blur = cv2.GaussianBlur(received, (0, 0), 1.0)
    return {
        "mse": np.mean(difference ** 2, axis=(1, 2, 3)).tolist(),
        "l1": np.mean(np.abs(difference), axis=(1, 2, 3)).tolist(),
        "blur_mse": [float(np.mean((cv2.GaussianBlur(x, (0, 0), 1.0).astype(np.float64)
                                    - target_blur) ** 2)) for x in images],
    }


def refine_bits(backend, codec, received, recovered, uncertain_bits=8, anchors=ANCHORS):
    """Return candidate predictions and inspectable receiver-only traces."""
    config = codec.config
    if (config.bits != 1 or config.repeat != 1 or config.ecc != "none" or config.whiten
            or codec.payload_size != config.channel_bits):
        raise ValueError("Prototype requires full-length uncoded, unwhitened one-bit mapping")
    if (not isinstance(uncertain_bits, int) or not 1 <= uncertain_bits <= codec.payload_size
            or not anchors or len(set(anchors)) != len(anchors) or any(x not in ANCHORS for x in anchors)):
        raise ValueError("Invalid search size or anchors")
    received = _unit(received)
    values = codec.recover_symbols(recovered)
    baseline = codec.decode(recovered)
    indices = np.argsort(np.abs(values.reshape(-1) - .5), kind="stable")[:uncertain_bits]
    predictions = {anchor + "_" + metric: baseline.copy() for anchor in anchors for metric in METRICS}
    traces = []
    for anchor in anchors:
        # For the hard anchor, the unchanged-message image is identical for
        # every tested position. K positions need K+1 generations, not 2K.
        hard_base = None
        if anchor == "hard":
            hard_base = backend.generate_many(
                symbols_to_latent(codec, (values > .5).astype(np.float64))[None], prompt="")[0]
        for rank, index in enumerate(indices):
            noises = candidate_pair(codec, values, int(index), anchor)
            # The sender's prompt and attack identity are deliberately unavailable.
            if anchor == "hard":
                original_bit = int(baseline[index])
                alternative = backend.generate_many(noises[[1 - original_bit]], prompt="")[0]
                images = np.stack([hard_base, alternative] if original_bit == 0 else [alternative, hard_base])
            else:
                images = backend.generate_many(noises, prompt="")
            distances = candidate_distances(images, received)
            choices = {}
            for metric, pair in distances.items():
                choice = int(baseline[index]) if abs(pair[0] - pair[1]) <= 1e-12 else int(pair[1] < pair[0])
                predictions[anchor + "_" + metric][index] = choice
                choices[metric] = choice
            traces.append({"anchor": anchor, "rank": rank, "index": int(index),
                           "symbol_estimate": float(values.flat[index]), "baseline_bit": int(baseline[index]),
                           "distances": distances, "choices": choices})
    return baseline, predictions, traces
