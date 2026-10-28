"""Observe the existing generator, then isolate receiver losses without caches.

Oracle intermediate latents and known prompts are diagnostic inputs ONLY.
The actual empty-prompt receiver path remains backend.invert, unchanged.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from grdh_eval.attacks import ATTACKS, AttackConfig, apply_attack, to_uint8
from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.data import select_prompts
from grdh_eval.runner import environment, write_json


def bit_transition(before, after, truth):
    before, after, truth = map(np.asarray, (before, after, truth))
    fixed = int(np.sum((before != truth) & (after == truth)))
    broken = int(np.sum((before == truth) & (after != truth)))
    return {"fixed": fixed, "broken": broken, "net_new_errors": broken - fixed}


def noise_metrics(codec, recovered, original, message):
    prediction = codec.decode(recovered)
    values = codec.recover_symbols(recovered).reshape(-1)
    margins = np.abs(values - .5)
    bins = [0, .025, .05, .1, .2, .5, float("inf")]
    return {"accuracy": float(np.mean(prediction == message)),
            "bit_errors": int(np.sum(prediction != message)),
            "noise_rmse": float(np.sqrt(np.mean((recovered.astype(float) - original) ** 2))),
            "confidence_bins": [{"lower": lo, "upper": hi if np.isfinite(hi) else None,
                                 "count": int(np.sum((margins >= lo) & (margins < hi))),
                                 "errors": int(np.sum((margins >= lo) & (margins < hi) & (prediction != message)))}
                                for lo, hi in zip(bins[:-1], bins[1:])]}, prediction


def observe_generation(backend, noise, prompt):
    """Capture actual VAE inputs/outputs; do not replace generation numerics."""
    captured = {}
    original_decode = backend.model.decode_first_stage

    def decode(latent, *args, **kwargs):
        captured["latent"] = latent.detach().clone()
        output = original_decode(latent, *args, **kwargs)
        unit = output[0].float().cpu().permute(1, 2, 0).numpy() / 2 + .5
        captured["unit"] = np.clip(unit, 0, 1)
        captured["clipped_fraction"] = float(np.mean((unit < 0) | (unit > 1)))
        return output

    backend.model.decode_first_stage = decode
    try:
        clean = backend.generate(noise, prompt)
    finally:
        backend.model.decode_first_stage = original_decode
    np.testing.assert_array_equal(to_uint8(captured["unit"]), clean)
    return clean, captured


def invert_internal(backend, latent, prompt):
    """Bypass the VAE using the true final diffusion latent (oracle diagnostic)."""
    import torch
    backend._start_phase("diagnostic internal inversion")
    with torch.no_grad(), backend.scope():
        c, uc = backend.conditioning(prompt, backend.config.inversion_guidance)
        recovered, _ = backend.sampler.sample(
            steps=backend.config.inversion_steps, batch_size=1, shape=tuple(latent.shape[1:]),
            x_T=latent.clone(), conditioning=c, unconditional_conditioning=uc,
            unconditional_guidance_scale=backend.config.inversion_guidance,
            order=backend.config.order, DPMencode=True, verbose=False)
    return recovered[0].float().cpu().numpy()


def aggregate(rows):
    if not rows:
        return {}
    result = {"images": len(rows), "stages": {}, "transitions": {},
              "note": "Paired net error changes are not independent causal contributions; fixed and broken bits are both counted."}
    for stage in rows[0]["stages"]:
        stats = [r["stages"][stage] for r in rows]
        result["stages"][stage] = {"accuracy": float(np.mean([s["accuracy"] for s in stats])),
                                  "bit_errors": sum(s["bit_errors"] for s in stats),
                                  "mean_noise_rmse": float(np.mean([s["noise_rmse"] for s in stats])),
                                  "confidence_bins": []}
        for i, entry in enumerate(stats[0]["confidence_bins"]):
            count = sum(s["confidence_bins"][i]["count"] for s in stats)
            errors = sum(s["confidence_bins"][i]["errors"] for s in stats)
            result["stages"][stage]["confidence_bins"].append(dict(entry, count=count, errors=errors,
                                                                         error_rate=errors / count if count else None))
    for transition in rows[0]["transitions"]:
        result["transitions"][transition] = {k: sum(r["transitions"][transition][k] for r in rows)
                                             for k in ("fixed", "broken", "net_new_errors")}
    result["compute_seconds"] = sum(r["seconds"] for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    from grdh_eval.backend import ModelConfig, StableDiffusionBackend
    import torch

    preset = json.loads(Path("configs/evaluation/optimized.json").read_text())["candidates"][0]
    config = ModelConfig(**preset["model"])
    options = dict(preset["codec"])
    options["shape"] = tuple(options["shape"])
    codec = SecretCodec(CodecConfig(**options))
    attacks = AttackConfig()
    prompts, prompt_hash = select_prompts("text_prompt_dataset/chatgpt_dataset.txt", "dev", args.count, 2026)
    manifest = json.loads(json.dumps({"model": asdict(config), "codec": asdict(codec.config),
                "attacks": asdict(attacks), "prompts": prompts, "prompts_sha256": prompt_hash,
                "seed": 2026, "count": args.count, "split": "dev",
                "oracle_inputs": "true internal diffusion latent / original prompt ONLY for labelled diagnostics",
                "storage": "aggregate and per-image numbers only; no images or latent files"}))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Changed experiment; use a new output")
    write_json(path, manifest)
    path = args.output / "per_image.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if [r["index"] for r in rows] != list(range(len(rows))) or len(rows) > args.count:
        raise ValueError("Invalid existing rows")
    if len(rows) == args.count:
        return 0
    backend = StableDiffusionBackend(config)
    write_json(args.output / "environment.json", dict(environment(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda))
    for index in range(len(rows), args.count):
        started = time.perf_counter()
        item = prompts[index]
        image_seed = (2026 + item["source_line"] * 1009) % 2**32
        message = np.random.RandomState(image_seed).randint(0, 2, codec.payload_size).astype(np.uint8)
        noise = codec.encode(message)
        clean, captured = observe_generation(backend, noise, item["prompt"])
        row = {"index": index, "source_line": item["source_line"], "payload_bits": len(message),
               "message_sha256": hashlib.sha256(message.tobytes()).hexdigest(),
               "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
               "clipped_fraction": captured["clipped_fraction"], "stages": {}, "transitions": {}}
        predictions = {}

        def record(name, recovered):
            row["stages"][name], predictions[name] = noise_metrics(codec, recovered, noise, message)
            print("STAGE", index, name, row["stages"][name]["accuracy"], flush=True)

        record("codec_exact", noise)
        for mode in ("known", "empty"):
            prompt = item["prompt"] if mode == "known" else ""
            record("internal_" + mode, invert_internal(backend, captured["latent"], prompt))
            backend.config = replace(config, prompt_mode=mode)
            inputs = {"float": captured["unit"], "clean": clean.astype(np.float32) / 255}
            inputs.update({attack: apply_attack(clean, attack, attacks, seed=image_seed + j)
                           for j, attack in enumerate(ATTACKS)})
            for stage, image in inputs.items():
                write_json(args.output / "status.json", {"index": index, "count": args.count, "stage": stage, "prompt": mode})
                record(stage + "_" + mode, backend.invert(image, prompt=prompt, seed=0))
            chain = ["codec_exact", "internal_" + mode, "float_" + mode, "clean_" + mode]
            pairs = list(zip(chain[:-1], chain[1:])) + [("clean_" + mode, a + "_" + mode) for a in ATTACKS]
            for before, after in pairs:
                row["transitions"][before + "->" + after] = bit_transition(predictions[before], predictions[after], message)
        for stage in ("internal", "float", "clean") + ATTACKS:
            row["transitions"][stage + "_known->" + stage + "_empty"] = bit_transition(
                predictions[stage + "_known"], predictions[stage + "_empty"], message)
        backend.config = config
        row["seconds"] = time.perf_counter() - started
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        rows.append(row)
        write_json(args.output / "summary.json", aggregate(rows))
    write_json(args.output / "status.json", {"status": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
