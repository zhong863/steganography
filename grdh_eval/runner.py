"""One generation, three independent attacked extractions, one slide-2 score."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .attacks import ATTACKS, apply_attack, to_uint8
from .data import select_prompts
from .quality import niqe_rgb
from .scoring import score


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python", "transformers", "pytorch-lightning"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def summarize(rows, planned_count, scope):
    result = {"status": "completed" if len(rows) == planned_count else "partial", "scope": scope,
              "images": len(rows), "planned_images": planned_count,
              "course_sample_count_met": len(rows) == 100 and scope in ("final", "ecc_final"),
              "official_equivalence": False,
              "protocol_note": "PPT formula; attack/NIQE conventions await TA confirmation."}
    if not rows:
        return result
    accuracies = {name: float(np.mean([row["acc_" + name] for row in rows])) for name in ATTACKS}
    mean_niqe = float(np.mean([row["niqe"] for row in rows]))
    bpp = float(np.mean([row["bpp"] for row in rows]))
    result.update(bpp=bpp, accuracies=accuracies, mean_niqe=mean_niqe,
                  score=score(bpp, list(accuracies.values()), mean_niqe).to_dict(),
                  compute_seconds=sum(row["seconds"] for row in rows))
    return result


def check_generation_source(source, current):
    """Only decoder changes may re-use a generated carrier image."""
    for field in ("prompts_sha256", "prompts", "seed", "model_file"):
        if source[field] != current[field]:
            raise ValueError("Generation cache differs in %s" % field)
    for field in ("checkpoint", "clip_dir", "config", "precision", "generation_steps", "order", "guidance", "attention", "image_size"):
        fallback = {"attention": "original", "image_size": 512}.get(field)
        if source["model"].get(field, fallback) != current["model"].get(field, fallback):
            raise ValueError("Generation cache differs in model.%s" % field)
    defaults = {"ecc": "none", "decode_gain": 1.0}
    left = {k: v for k, v in dict(defaults, **source["codec"]).items() if k not in ("decision", "decode_gain")}
    right = {k: v for k, v in dict(defaults, **current["codec"]).items() if k not in ("decision", "decode_gain")}
    if left != right:
        raise ValueError("Generation cache encoding differs; new carriers must be generated")


def run_evaluation(backend, codec_config, attack_config, prompts, output, split="dev", count=None,
                   seed=2026, niqe_color="y", resume=False, generation_source=None, stop_file=None):
    from .backend import StegoMethod

    selected, prompts_sha = select_prompts(prompts, split, count, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(backend.config.checkpoint)
    model_identity = {"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size,
                      "mtime_ns": checkpoint.stat().st_mtime_ns}
    manifest = {"schema": 1, "codec": asdict(codec_config), "model": asdict(backend.config),
                "model_file": model_identity, "attacks": asdict(attack_config),
                "niqe": {"implementation": "vendored BasicSR", "convert_to": niqe_color, "crop_border": 0},
                "prompts_sha256": prompts_sha, "prompts": selected, "split": split, "seed": seed,
                "generation_source": str(Path(generation_source).resolve()) if generation_source else None,
                "image_format": "uint8 RGB PNG; truncation; AWGN extracted from float32 unit RGB",
                "payload_framing": "fixed length from shared configuration; no per-image length side channel"}
    # Normalize tuples through JSON so the resume comparison is exact.
    manifest = json.loads(json.dumps(manifest))
    if generation_source:
        source_manifest = json.loads((Path(generation_source) / "manifest.json").read_text(encoding="utf-8"))
        check_generation_source(source_manifest, manifest)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already exists; choose a new directory or use --resume")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous.setdefault("generation_source", None)
        previous["model"].setdefault("attention", "original")
        previous["model"].setdefault("image_size", 512)
        previous["model"].setdefault("receiver_filter", "none")
        previous["codec"].setdefault("ecc", "none")
        previous["codec"].setdefault("decode_gain", 1.0)
        if previous != manifest:
            raise ValueError("Resume settings or input files do not match the original manifest")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty output directory")
        write_json(manifest_path, manifest)
        write_json(output / "environment.json", environment())
    for name in ("clean", "attacked", "latents", "messages"):
        (output / name).mkdir(exist_ok=True)
    rows_path = output / "per_image.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()] if rows_path.exists() else []
    done = {row["index"] for row in rows}
    if len(done) != len(rows) or any(i < 0 or i >= len(selected) for i in done):
        raise ValueError("Invalid or duplicated saved image records")
    method = StegoMethod(backend, codec_config)
    if codec_config.shape != (4, backend.config.image_size // 8, backend.config.image_size // 8):
        raise ValueError("Codec shape must match the configured image size")
    try:
        for index, item in enumerate(selected):
            if index in done:
                continue
            if stop_file is not None and Path(stop_file).exists():
                raise KeyboardInterrupt("Stop file requested; completed images are saved")
            started = time.perf_counter()
            identifier = "%03d" % index
            image_seed = (seed + item["source_line"] * 1009) % (2 ** 32)
            bits = np.random.RandomState(image_seed).randint(0, 2, method.codec.payload_size).astype(np.uint8)
            print("[%d/%d] generate (%s)" % (index + 1, len(selected), backend.config.prompt_mode), flush=True)
            clean_path = output / "clean" / (identifier + ".png")
            cached = Path(generation_source) / "clean" / (identifier + ".png") if generation_source else None
            if cached and cached.is_file() and not clean_path.exists():
                reference_bits = np.load(Path(generation_source) / "messages" / (identifier + ".npy"), allow_pickle=False)
                if not np.array_equal(reference_bits, bits):
                    raise ValueError("Cached generation used a different secret message")
                shutil.copyfile(cached, clean_path)
            if (resume or generation_source) and clean_path.is_file():
                with Image.open(clean_path) as image:
                    clean = np.array(image.convert("RGB"))
            else:
                clean = method.embed(bits, item["prompt"])
                Image.fromarray(clean).save(clean_path)
            np.save(output / "messages" / (identifier + ".npy"), bits)
            row = {"index": index, "source_line": item["source_line"], "payload_bits": int(bits.size),
                   "generation_reused": bool(cached and cached.is_file()),
                   "bpp": bits.size / (clean.shape[0] * clean.shape[1]),
                   "niqe": niqe_rgb(clean, niqe_color)}
            # Only the known-prompt diagnostic is given the original prompt.
            shared_prompt = item["prompt"] if backend.config.prompt_mode == "known" else None
            for number, attack in enumerate(ATTACKS):
                print("[%d/%d] invert %s" % (index + 1, len(selected), attack), flush=True)
                received = apply_attack(clean, attack, attack_config, seed=image_seed + number)
                Image.fromarray(to_uint8(received)).save(output / "attacked" / (identifier + "_" + attack + ".png"))
                if attack == "awgn":
                    np.save(output / "attacked" / (identifier + "_awgn_exact.npy"), received)
                latent_path = output / "latents" / (identifier + "_" + attack + ".npy")
                if resume and latent_path.is_file():
                    recovered = np.load(latent_path, allow_pickle=False)
                else:
                    recovered = backend.invert(received, prompt=shared_prompt, seed=image_seed + number)
                    np.save(latent_path, recovered)
                prediction = method.codec.decode(recovered)
                row["acc_" + attack] = float(np.mean(prediction == bits))
            row["seconds"] = time.perf_counter() - started
            with rows_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            rows.append(row)
            write_json(output / "summary.json", summarize(rows, len(selected), split))
            print("[%d/%d] %s" % (index + 1, len(selected), json.dumps(row)), flush=True)
    except BaseException as error:
        write_json(output / "summary.json", dict(summarize(rows, len(selected), split), error=str(error)))
        raise
    summary = summarize(rows, len(selected), split)
    write_json(output / "summary.json", summary)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def redecode(run_dir, decision="soft"):
    """Change ONLY the decision rule, using the exact cached attacked latents."""
    from .codec import CodecConfig, SecretCodec

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    options = dict(manifest["codec"], decision=decision)
    options["shape"] = tuple(options["shape"])
    codec = SecretCodec(CodecConfig(**options))
    originals = [json.loads(line) for line in (root / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for original in originals:
        row = dict(original)
        identifier = "%03d" % row["index"]
        bits = np.load(root / "messages" / (identifier + ".npy"), allow_pickle=False)
        for attack in ATTACKS:
            latent = np.load(root / "latents" / (identifier + "_" + attack + ".npy"), allow_pickle=False)
            row["acc_" + attack] = float(np.mean(codec.decode(latent) == bits))
        rows.append(row)
    summary = summarize(rows, len(manifest["prompts"]), manifest["split"])
    summary["redecoded_with"] = decision
    write_json(root / ("redecoded_" + decision + ".json"), summary)
    return summary
