"""Inspect cached inversion errors to guide later carrier/band experiments."""

import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from .attacks import ATTACKS
from .codec import CodecConfig, SecretCodec
from .runner import write_json


def diagnose(run):
    root = Path(run)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    settings = dict(manifest["codec"])
    settings["shape"] = tuple(settings["shape"])
    codec = SecretCodec(CodecConfig(**settings))
    records = [json.loads(line) for line in (root / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("No completed images to diagnose")
    height, width = codec.config.shape[-2:]
    y, x = np.meshgrid(np.arange(height) / height, np.arange(width) / width, indexing="ij")
    radius = np.sqrt((x ** 2 + y ** 2) / 2)
    bands = {"low": radius < .25, "middle": (radius >= .25) & (radius < .6), "high": radius >= .6}
    output = {}
    for attack in ATTACKS:
        errors, signals = [], []
        for row in records:
            identifier = "%03d" % row["index"]
            message = np.load(root / "messages" / (identifier + ".npy"), allow_pickle=False)
            original = codec.encode(message).astype(np.float64)
            recovered = np.load(root / "latents" / (identifier + "_" + attack + ".npy"), allow_pickle=False)
            errors.append(recovered - original)
            signals.append(original)
        error, signal = np.stack(errors), np.stack(signals)
        spectrum_error = dctn(error, axes=(-2, -1), norm="ortho")
        spectrum_signal = dctn(signal, axes=(-2, -1), norm="ortho")
        frequency = {}
        for name, mask in bands.items():
            mse = float(np.mean(spectrum_error[..., mask] ** 2))
            power = float(np.mean(spectrum_signal[..., mask] ** 2))
            frequency[name] = {"error_mse": mse, "signal_power": power,
                               "snr_db": float(10 * np.log10(max(power, 1e-30) / max(mse, 1e-30)))}
        output[attack] = {"latent_mae": float(np.mean(np.abs(error))),
                          "latent_mse": float(np.mean(error ** 2)),
                          "mse_by_channel": np.mean(error ** 2, axis=(0, 2, 3)).tolist(),
                          "dct_bands": frequency}
    report = {"images": len(records), "kind": "inversion_error_diagnostic", "attacks": output,
              "note": "Diagnostic only. DCT basis changes alone do not establish a robustness improvement."}
    write_json(root / "diagnostics.json", report)
    return report
