"""Small-disk, fixed-protocol GPU experiments driven by a JSON plan.

Only clean carriers and (optionally) compressed recovered latents are retained.
Attacked images are transient. The receiver never receives payloads or attack names.
"""

import argparse
import gc
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
from PIL import Image

from grdh_eval.attacks import ATTACKS, AttackConfig, apply_attack
from grdh_eval.codec import CodecConfig, SecretCodec, symbols_to_bits
from grdh_eval.data import select_prompts
from grdh_eval.quality import niqe_rgb
from grdh_eval.runner import environment, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Plan changed; choose a new output directory")
    write_json(plan_path, plan)
    from grdh_eval.backend import ModelConfig, StableDiffusionBackend

    base = ModelConfig(checkpoint="models/sd15/author-nonema-inference.safetensors",
                       device="cuda:0", precision="autocast", attention="original",
                       posterior="mode", prompt_mode="empty")
    base = replace(base, **plan.get("model", {}))
    attacks = AttackConfig()  # Same fixed course interpretation across all candidates.
    seed = plan.get("seed", 2026)
    message_seed_offset = plan.get("message_seed_offset", 0)
    split = plan.get("split", "dev")
    count = plan.get("count", 5)
    keep_clean = plan.get("keep_clean", count)
    if not isinstance(keep_clean, int) or not 0 <= keep_clean <= count:
        raise ValueError("keep_clean must be between zero and the image count")
    prompts, prompt_hash = select_prompts("text_prompt_dataset/chatgpt_dataset.txt", split, count, seed)
    backend = None
    summaries = {}
    for spec in plan["candidates"]:
        name = spec["name"]
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("Invalid candidate name")
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        config = replace(base, **spec.get("model", {}))
        codec_options = dict(spec.get("codec", {}))
        if "shape" in codec_options:
            codec_options["shape"] = tuple(codec_options["shape"])
        codec = SecretCodec(CodecConfig(**codec_options))
        cache_latents = spec.get("cache_latents", plan.get("cache_latents", False))
        manifest = {"schema": 1, "model": asdict(config), "codec": asdict(codec.config),
                    "attacks": asdict(attacks), "prompts": prompts, "prompts_sha256": prompt_hash,
                    "seed": seed, "split": split, "count": count,
                    "message_seed_offset": message_seed_offset,
                    "diagnose_channel": plan.get("diagnose_channel", False),
                    "niqe": "BasicSR Y, clean uint8 image; average before score",
                    "receiver_seed": 0, "receiver_inputs": "attacked image only plus fixed configuration",
                    "cache_latents": cache_latents, "keep_clean": keep_clean}
        manifest = json.loads(json.dumps(manifest))
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            previous.setdefault("keep_clean", count)
            previous.setdefault("message_seed_offset", 0)
            previous.setdefault("diagnose_channel", False)
            previous["codec"].setdefault("ecc", "none")
            previous["codec"].setdefault("decode_gain", 1.0)
            previous["model"].setdefault("receiver_filter", "none")
            if previous != manifest:
                raise ValueError("Candidate changed; choose a new destination")
        write_json(manifest_path, manifest)
        rows_path = destination / "per_image.jsonl"
        rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
        if [r["index"] for r in rows] != list(range(len(rows))) or len(rows) > count:
            raise ValueError("Invalid saved records")
        if len(rows) < count:
            if backend is None:
                backend = StableDiffusionBackend(config)
                import torch
                write_json(args.output / "environment.json", dict(environment(),
                           gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda))
            elif any(getattr(config, key) != getattr(backend.config, key)
                     for key in ("checkpoint", "config", "clip_dir", "device", "attention")):
                # Release the old model before loading another existing checkpoint.
                del backend
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                backend = StableDiffusionBackend(config)
            backend.config = config
        for index in range(len(rows), count):
            if (args.output / "STOP").exists():
                return 0
            item = prompts[index]
            write_json(args.output / "status.json", {"candidate": name, "index": index, "count": count})
            started = time.perf_counter()
            image_seed = (seed + item["source_line"] * 1009 + message_seed_offset) % 2**32
            message = np.random.RandomState(image_seed).randint(0, 2, codec.payload_size).astype(np.uint8)
            clean_path = destination / ("%03d.png" % index)
            source = spec.get("reuse_clean")
            if source:
                # Verify every encoder/generator input before using another carrier.
                source = Path(source)
                other = json.loads((source / "manifest.json").read_text())
                if other.get("message_seed_offset", 0) != message_seed_offset:
                    raise ValueError("Reuse changes message seed offset")
                for key in ("prompts", "seed"):
                    if other[key] != manifest[key]:
                        raise ValueError("Reuse changes " + key)
                for key in ("checkpoint", "clip_dir", "config", "precision", "generation_steps", "order", "guidance", "attention", "image_size"):
                    if other["model"][key] != manifest["model"][key]:
                        raise ValueError("Reuse changes generation " + key)
                for key, value in manifest["codec"].items():
                    if key not in ("decision", "decode_gain") and other["codec"].get(key, asdict(CodecConfig()).get(key)) != value:
                        raise ValueError("Reuse changes codec " + key)
                source_path = source / ("%03d.png" % index)
                if not source_path.exists():
                    source_path = source / "clean" / ("%03d.png" % index)
                if source_path.exists():
                    with Image.open(source_path) as image:
                        clean = np.array(image.convert("RGB"))
                else:
                    # Compact runs may deliberately remove image caches later.
                    # Verified identical encoder inputs are sufficient to regenerate.
                    clean = backend.generate(codec.encode(message), item["prompt"])
            elif clean_path.exists():
                with Image.open(clean_path) as image:
                    clean = np.array(image.convert("RGB"))
            else:
                clean = backend.generate(codec.encode(message), item["prompt"])
            if index < keep_clean and not clean_path.exists():
                Image.fromarray(clean).save(clean_path)
            row = {"index": index, "source_line": item["source_line"], "payload_bits": len(message),
                   "bpp": len(message) / (512 * 512), "niqe": niqe_rgb(clean),
                   "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                   "message_sha256": hashlib.sha256(message.tobytes()).hexdigest()}
            if plan.get("diagnose_channel", False):
                # Evaluator-only measurement of uncoded channel errors on this
                # SAME carrier. Includes parity/padding; never used for scoring.
                truth_symbols = np.rint(codec.recover_symbols(codec.encode(message))).astype(np.int64)
                channel_truth = symbols_to_bits(truth_symbols, codec.config.bits)
            cache = {}
            for j, attack in enumerate(ATTACKS):
                received = apply_attack(clean, attack, attacks, seed=image_seed + j)
                recovered = backend.invert(received, prompt=item["prompt"] if config.prompt_mode == "known" else None, seed=0)
                prediction = codec.decode(recovered)
                row["acc_" + attack] = float(np.mean(prediction == message))
                if plan.get("diagnose_channel", False):
                    raw_symbols = np.clip(np.rint(codec.recover_symbols(recovered)), 0,
                                          2 ** codec.config.bits - 1).astype(np.int64)
                    row["raw_channel_acc_" + attack] = float(np.mean(
                        symbols_to_bits(raw_symbols, codec.config.bits) == channel_truth))
                row["latent_std_" + attack] = float(recovered.std())
                cache[attack] = recovered
            if cache_latents:
                np.savez_compressed(destination / ("%03d_latents.npz" % index), **cache)
            row["seconds"] = time.perf_counter() - started
            with rows_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            rows.append(row)
            summary = summarize(rows, count, split)
            write_json(destination / "summary.json", summary)
            print("RESULT", name, index + 1, json.dumps(summary), flush=True)
        summaries[name] = summarize(rows, count, split)
        write_json(args.output / "summaries.json", summaries)
    write_json(args.output / "status.json", {"status": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
