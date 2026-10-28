"""Audit the error decomposition / ECC experiment and score paired holdouts."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from grdh_eval.runner import write_json
from scripts.summarize_gpu_study import paired_bootstrap


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/error_ecc"))
    args = parser.parse_args()
    root = args.root
    report = {"decomposition": json.loads((root / "decomposition/summary.json").read_text()),
              "stages": {}, "comparisons": {}, "audit": {}}
    table = []
    for stage in ("screen", "calibration", "validation", "final"):
        folders = sorted((root / stage).glob("*/summary.json"))
        if not folders:
            continue
        reports = {}
        for path in folders:
            s = json.loads(path.read_text())
            if s["status"] != "completed":
                raise ValueError("Incomplete study: " + str(path))
            reports[path.parent.name] = s
            manifest = json.loads((path.parent / "manifest.json").read_text())
            for row in read_rows(path.parent / "per_image.jsonl"):
                if row["bpp"] != row["payload_bits"] / (512 * 512):
                    raise ValueError("Effective payload capacity is inconsistent")
                seed = (manifest["seed"] + row["source_line"] * 1009 + manifest.get("message_seed_offset", 0)) % 2**32
                message = np.random.RandomState(seed).randint(0, 2, row["payload_bits"]).astype(np.uint8)
                if hashlib.sha256(message.tobytes()).hexdigest() != row["message_sha256"]:
                    raise ValueError("Message seed/prefix identity does not match")
            table.append(dict(stage=stage, candidate=path.parent.name, images=s["images"], bpp=s["bpp"],
                              niqe=s["mean_niqe"], **s["accuracies"], **s["score"], seconds=s["compute_seconds"]))
        report["stages"][stage] = reports
        if stage not in ("validation", "final"):
            continue
        reference = root / stage / "uncoded"
        base_manifest = json.loads((reference / "manifest.json").read_text())
        base_rows = read_rows(reference / "per_image.jsonl")
        report["comparisons"][stage] = {}
        for name in reports:
            manifest = json.loads((root / stage / name / "manifest.json").read_text())
            for key in ("model", "attacks", "prompts", "prompts_sha256", "seed", "message_seed_offset", "niqe", "count", "diagnose_channel"):
                if base_manifest[key] != manifest[key]:
                    raise ValueError("Paired protocol differs in " + key)
            rows = read_rows(root / stage / name / "per_image.jsonl")
            report["comparisons"][stage][name] = paired_bootstrap(base_rows, rows)
            report["comparisons"][stage][name]["score_component_differences"] = {
                key: reports[name]["score"][key] - reports["uncoded"]["score"][key]
                for key in ("capacity", "robustness", "quality", "total")}
            if all("raw_channel_acc_jpeg" in r for r in rows):
                reports[name]["raw_channel_accuracies"] = {
                    attack: sum(r["raw_channel_acc_" + attack] for r in rows) / len(rows)
                    for attack in ("jpeg", "awgn", "mblur")}
                reports[name]["raw_channel_note"] = "Before ECC; includes parity and padding, so it is diagnostic only, not course payload accuracy."
                reports[name]["complete_payload_recovery_counts"] = {
                    attack: sum(r["acc_" + attack] == 1 for r in rows) for attack in ("jpeg", "awgn", "mblur")}
                reports[name]["complete_payload_under_all_three_attacks"] = sum(
                    all(r["acc_" + attack] == 1 for attack in ("jpeg", "awgn", "mblur")) for r in rows)
                reports[name]["images_below_90_percent_payload_accuracy"] = {
                    attack: sum(r["acc_" + attack] < .9 for r in rows) for attack in ("jpeg", "awgn", "mblur")}
        selection_path = root / ("development_selection.json" if stage == "validation" else "validation_selection.json")
        selection = json.loads(selection_path.read_text())
        plan = json.loads((root / stage / "plan.json").read_text())
        if plan != selection["frozen_plan"]:
            raise ValueError("Run plan differs from frozen selection")
        report["audit"][stage + "_matches_frozen_plan"] = True
    # Oracle diagnostics and regular evaluation must produce the same carriers.
    diagnostic_rows = read_rows(root / "decomposition/per_image.jsonl")
    base_rows = read_rows(root / "screen/uncoded/per_image.jsonl")
    if len(diagnostic_rows) != len(base_rows):
        raise ValueError("Decomposition/screen sample counts differ")
    for diag, base in zip(diagnostic_rows, base_rows):
        for key in ("index", "source_line", "message_sha256", "clean_sha256"):
            if diag[key] != base[key]:
                raise ValueError("Instrumentation changed " + key)
        for attack in ("jpeg", "awgn", "mblur"):
            if diag["stages"][attack + "_empty"]["accuracy"] != base["acc_" + attack]:
                raise ValueError("Instrumentation changed receiver output")
    report["audit"]["diagnostics_match_regular_receiver_images"] = len(base_rows)
    prompt_sets = []
    for stage in ("screen", "validation", "final"):
        path = root / stage / "uncoded/manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            prompt_sets.append({x["prompt"] for x in manifest["prompts"]})
    if any(left & right for i, left in enumerate(prompt_sets) for right in prompt_sets[i + 1:]):
        raise ValueError("Current study splits overlap")
    historical = json.loads(Path("results/optimization/final/optimized/manifest.json").read_text())
    previous_prompts = {x["prompt"] for x in historical["prompts"]}
    if any(group & previous_prompts for group in prompt_sets[1:]):
        raise ValueError("New holdout overlaps the previous final test")
    report["audit"]["new_holdouts_disjoint_from_previous_final"] = True
    # Use the frozen inventory, not mutable future results directories.
    inventory = json.loads((root / "historical_prompt_inventory.json").read_text())
    historical_prompts = set(inventory["prompts"])
    if any(group & historical_prompts for group in prompt_sets[1:]):
        raise ValueError("New holdout appears in another historical experiment")
    report["audit"]["new_holdouts_disjoint_from_all_historical_manifests"] = True
    report["audit"]["historical_unique_prompts"] = len(historical_prompts)
    report["audit"]["message_seed_hashes_and_effective_payload_capacity_verified"] = True
    report["generation_image_evaluations"] = len(diagnostic_rows) + sum(
        value["images"] for stage, candidates in report["stages"].items() if stage != "calibration"
        for value in candidates.values())
    report["decoder_only_calibration_image_evaluations"] = sum(
        value["images"] for value in report["stages"].get("calibration", {}).values())
    report["independent_prompts_this_study"] = sum(map(len, prompt_sets))
    report["storage_note"] = "Calibration latents are regenerable temporary caches; no attacked images are stored."
    write_json(root / "report.json", report)
    with (root / "all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({"comparisons": report["comparisons"], "audit": report["audit"]}, indent=2))


if __name__ == "__main__":
    main()
