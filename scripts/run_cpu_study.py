"""Resumable CPU candidate screening, validation, and 100-prompt evaluation."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grdh_eval.attacks import AttackConfig
from grdh_eval.codec import CodecConfig
from grdh_eval.data import select_prompts
from grdh_eval.runner import run_evaluation, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/cpu_study"))
    p.add_argument("--checkpoint", default="models/sd15/v1-5-pruned-emaonly.safetensors")
    p.add_argument("--clip-dir", default="models/sd15/clip")
    p.add_argument("--prompts", default="text_prompt_dataset/chatgpt_dataset.txt")
    p.add_argument("--dev-count", type=int, default=5)
    p.add_argument("--validation-count", type=int, default=20)
    p.add_argument("--final-count", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--attention", choices=("original", "sdpa"), default="sdpa")
    p.add_argument("--until", choices=("dev", "validation", "final"), default="final")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stop = args.output / "STOP"
    status_path = args.output / "status.json"
    write_json(args.output / "process.json", {"pid": os.getpid(), "command": sys.argv})
    if stop.exists():
        write_json(status_path, {"status": "stopped", "reason": "Remove STOP before resuming"})
        return 0
    from grdh_eval.backend import ModelConfig, StableDiffusionBackend

    model = ModelConfig(checkpoint=args.checkpoint, clip_dir=args.clip_dir, device="cpu",
                        threads=args.threads, attention=args.attention, posterior="mode", prompt_mode="empty")
    candidates = {
        "reference": CodecConfig(decision="hard"),
        "repeat2": CodecConfig(repeat=2, whiten=True),
        "repeat4": CodecConfig(repeat=4, whiten=True),
        "multibit2": CodecConfig(bits=2, whiten=True),
        "multibit2_repeat2": CodecConfig(bits=2, repeat=2, whiten=True),
    }
    for split, count in (("dev", args.dev_count), ("validation", args.validation_count), ("final", args.final_count)):
        select_prompts(args.prompts, split, count, args.seed)
    plan = {"model": asdict(model), "candidates": {k: asdict(v) for k, v in candidates.items()},
            "prompts": args.prompts, "seed": args.seed,
            "counts": {"dev": args.dev_count, "validation": args.validation_count, "final": args.final_count},
            "attacks": asdict(AttackConfig()), "selection": "top two dev scores; best validation score; final untouched split"}
    plan = json.loads(json.dumps(plan))
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Study settings changed; use a new output directory")
    write_json(plan_path, plan)
    started = time.monotonic()
    completed = {}
    try:
        write_json(status_path, {"status": "loading_model", "pid": os.getpid()})
        backend = StableDiffusionBackend(model)

        def run_stage(split, names, count):
            reports = {}
            for name in names:
                if stop.exists():
                    raise KeyboardInterrupt("Stop requested")
                write_json(status_path, {"status": "running", "stage": split, "candidate": name,
                                        "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started,
                                        "progress_file": str(args.output / split / name / "summary.json")})
                destination = args.output / split / name
                reports[name] = run_evaluation(
                    backend, candidates[name], AttackConfig(), args.prompts, destination,
                    split=split, count=count, seed=args.seed,
                    resume=(destination / "manifest.json").exists(), stop_file=stop)
                write_json(args.output / (split + "_comparison.json"), reports)
            completed[split] = reports
            return sorted(reports, key=lambda n: reports[n]["score"]["total"], reverse=True)

        top = run_stage("dev", list(candidates), args.dev_count)
        if args.until == "dev":
            write_json(status_path, {"status": "completed_requested_stage", "stage": "dev", "ranking": top})
            return 0
        top = run_stage("validation", top[:2], args.validation_count)
        if args.until == "validation":
            write_json(status_path, {"status": "completed_requested_stage", "stage": "validation", "ranking": top})
            return 0
        winner = top[0]
        run_stage("final", [winner], args.final_count)
        write_json(args.output / "final_result.json", {
            "winner": winner, "codec": asdict(candidates[winner]),
            "validation": completed["validation"][winner], "final": completed["final"][winner],
            "note": "Local PPT protocol; TA implementation equivalence not established."})
        write_json(status_path, {"status": "completed", "winner": winner,
                                "elapsed_seconds": time.monotonic() - started,
                                "final_result": str(args.output / "final_result.json")})
    except KeyboardInterrupt as error:
        write_json(status_path, {"status": "stopped", "reason": str(error), "resume": "Rerun the same command after removing STOP"})
        return 130
    except Exception as error:
        write_json(status_path, {"status": "failed", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
