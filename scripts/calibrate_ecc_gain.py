"""Reuse temporary Hamming latents to screen decoder gains without GPU work.

Only development data is accepted. This performs model selection, not a final
test; the resulting configuration must subsequently be frozen and validated.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from grdh_eval.attacks import ATTACKS
from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.runner import summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gains", type=float, nargs="+", default=[.25, .5, 2, 4])
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest["split"] != "dev" or manifest["codec"]["ecc"] != "hamming15":
        raise ValueError("Calibration requires development Hamming data")
    originals = [json.loads(x) for x in (args.source / "per_image.jsonl").read_text().splitlines()]
    if len(originals) != manifest["count"]:
        raise ValueError("Complete the source experiment first")
    for gain in args.gains:
        options = dict(manifest["codec"], decode_gain=gain)
        options["shape"] = tuple(options["shape"])
        codec = SecretCodec(CodecConfig(**options))
        destination = args.output / (args.source.name + "_gain" + str(gain).replace(".", "p"))
        destination.mkdir(parents=True, exist_ok=True)
        spec = dict(manifest, codec=options, generation_source=str(args.source), cache_latents=False,
                    keep_clean=0, timing_note="Source generation/inversion/default-decode time plus recalibration CPU time; conservative overestimate.")
        spec = json.loads(json.dumps(spec))
        path = destination / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError("Calibration configuration changed")
        write_json(path, spec)
        rows = []
        for original in originals:
            row = dict(original)
            started = time.perf_counter()
            seed = (manifest["seed"] + row["source_line"] * 1009 + manifest.get("message_seed_offset", 0)) % 2**32
            truth = np.random.RandomState(seed).randint(0, 2, codec.payload_size).astype(np.uint8)
            with np.load(args.source / ("%03d_latents.npz" % row["index"])) as cache:
                for attack in ATTACKS:
                    # Truth is used only after the decoder has returned its bits.
                    prediction = codec.decode(cache[attack])
                    row["acc_" + attack] = float(np.mean(prediction == truth))
            row["calibration_seconds"] = time.perf_counter() - started
            row["seconds"] += row["calibration_seconds"]
            rows.append(row)
        (destination / "per_image.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = summarize(rows, manifest["count"], "dev")
        write_json(destination / "summary.json", report)
        print(destination.name, report["score"]["total"], flush=True)


if __name__ == "__main__":
    main()
