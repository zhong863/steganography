"""Paired uncertain-bit reconstruction experiment with tiny disk footprint."""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from grdh_eval.attacks import ATTACKS, AttackConfig, apply_attack
from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.data import select_prompts
from grdh_eval.quality import niqe_rgb
from grdh_eval.regenerate import ANCHORS, METRICS, refine_bits
from grdh_eval.runner import environment, summarize, write_json


def reports(rows, count, split, names):
    result = {}
    for name in names:
        ordinary = [dict(r, **{"acc_" + a: r["variants"][name]["acc_" + a] for a in ATTACKS}) for r in rows]
        summary = summarize(ordinary, count, split)
        if rows:
            summary["search_counts"] = {field: sum(r["variants"][name].get(field, 0) for r in rows)
                                        for field in ("selected", "selected_before_correct", "selected_after_correct", "fixed", "broken", "flipped")}
            search_seconds = sum(r["search_seconds_" + a] for r in rows for a in ATTACKS)
            summary["all_variants_study_seconds"] = summary.pop("compute_seconds")
            summary["baseline_seconds"] = summary["all_variants_study_seconds"] - search_seconds
            summary["all_variants_search_seconds"] = search_seconds
            summary["timing_note"] = "Search time includes all tested anchors; distance metrics reuse the same candidate images."
        result[name] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "validation"), default="dev")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--uncertain-bits", type=int, default=8)
    parser.add_argument("--anchors", nargs="+", choices=ANCHORS, default=list(ANCHORS))
    parser.add_argument("--message-seed-offset", type=int, default=0)
    args = parser.parse_args()
    from grdh_eval.backend import ModelConfig, StableDiffusionBackend

    settings = json.loads(Path("configs/evaluation/optimized.json").read_text())["candidates"][0]
    model = ModelConfig(**settings["model"])
    options = dict(settings["codec"])
    options["shape"] = tuple(options["shape"])
    codec = SecretCodec(CodecConfig(**options))
    attacks = AttackConfig()
    selected, prompt_sha = select_prompts("text_prompt_dataset/chatgpt_dataset.txt", args.split, args.count, 2026)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"model": asdict(model), "codec": asdict(codec.config), "attacks": asdict(attacks),
                "split": args.split, "prompts": selected, "prompts_sha256": prompt_sha, "seed": 2026,
                "message_seed_offset": args.message_seed_offset, "uncertain_bits": args.uncertain_bits,
                "anchors": args.anchors, "metrics": METRICS, "receiver_prompt": "",
                "receiver_inputs": "received image and its recovered noise only; no true message, clean image or attack label",
                "search": "independent bit hypotheses using frozen initial estimate",
                "storage": "no candidate images or latent caches saved; first clean image only"}
    manifest = json.loads(json.dumps(manifest))
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Experiment changed; choose a new output directory")
    write_json(path, manifest)
    path = args.output / "per_image.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if [r["index"] for r in rows] != list(range(len(rows))) or len(rows) > args.count:
        raise ValueError("Invalid saved rows")
    names = ["baseline"] + [a + "_" + m for a in args.anchors for m in METRICS]
    if len(rows) == args.count:
        return 0
    backend = StableDiffusionBackend(model)
    import torch
    write_json(args.output / "environment.json", dict(environment(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
               hard_common_image_cache=True,
               candidate_generations_per_extraction={a: args.uncertain_bits + 1 if a == "hard" else 2 * args.uncertain_bits
                                                    for a in args.anchors}))
    for index in range(len(rows), args.count):
        item = selected[index]
        started = time.perf_counter()
        image_seed = (2026 + item["source_line"] * 1009 + args.message_seed_offset) % 2**32
        message = np.random.RandomState(image_seed).randint(0, 2, codec.payload_size).astype(np.uint8)
        clean = backend.generate(codec.encode(message), item["prompt"])
        if index == 0:
            Image.fromarray(clean).save(args.output / "example.png")
        row = {"index": index, "source_line": item["source_line"], "bpp": len(message) / (512 * 512),
               "payload_bits": len(message), "niqe": niqe_rgb(clean),
               "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
               "message_sha256": hashlib.sha256(message.tobytes()).hexdigest(),
               "variants": {name: {field: 0 for field in ("selected", "selected_before_correct", "selected_after_correct", "fixed", "broken", "flipped")} for name in names}}
        for number, attack in enumerate(ATTACKS):
            write_json(args.output / "status.json", {"status": "running", "index": index, "count": args.count, "attack": attack})
            received = apply_attack(clean, attack, attacks, seed=image_seed + number)
            recovered = backend.invert(received, seed=0)
            search_started = time.perf_counter()
            before, predictions, traces = refine_bits(backend, codec, received, recovered, args.uncertain_bits, args.anchors)
            row["search_seconds_" + attack] = time.perf_counter() - search_started
            indices = np.array([t["index"] for t in traces if t["anchor"] == args.anchors[0]])
            # The evaluator uses truth AFTER receiver decisions, never during ranking.
            for name, prediction in dict(predictions, baseline=before).items():
                v = row["variants"][name]
                v["acc_" + attack] = float(np.mean(prediction == message))
                v["selected"] += len(indices)
                v["selected_before_correct"] += int(np.sum(before[indices] == message[indices]))
                v["selected_after_correct"] += int(np.sum(prediction[indices] == message[indices]))
                v["fixed"] += int(np.sum((before != message) & (prediction == message)))
                v["broken"] += int(np.sum((before == message) & (prediction != message)))
                v["flipped"] += int(np.sum(before != prediction))
            write_json(args.output / ("%03d_%s_decisions.json" % (index, attack)), traces)
            print("ATTACK_RESULT", index, attack, json.dumps({n: row["variants"][n]["acc_" + attack] for n in names}), flush=True)
        row["seconds"] = time.perf_counter() - started
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        rows.append(row)
        report = reports(rows, args.count, args.split, names)
        write_json(args.output / "summary.json", report)
        print("IMAGE_RESULT", index, json.dumps({n: r["score"]["total"] for n, r in report.items()}), flush=True)
    write_json(args.output / "status.json", {"status": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
