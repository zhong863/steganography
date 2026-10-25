"""Run with python -m grdh_eval --help from the repository root."""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="PPT-matched scoring and training-free GRDH experiments")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect runtime and local model assets without loading SD")
    s = commands.add_parser("score", help="Calculate the exact slide-2 formula")
    s.add_argument("--bpp", type=float, required=True)
    s.add_argument("--accuracies", type=float, nargs=3, required=True, metavar=("JPEG", "AWGN", "MEDIAN"))
    s.add_argument("--niqe", type=float, required=True)
    sim = commands.add_parser("simulate", help="Latent-channel codec experiment; NOT an image benchmark or course score")
    sim.add_argument("--trials", type=int, default=20)
    sim.add_argument("--output", type=Path, default=Path("results/latent_channel.json"))
    red = commands.add_parser("redecode", help="Re-use saved latents to compare hard and soft decoding")
    red.add_argument("run", type=Path)
    red.add_argument("--decision", choices=("hard", "soft"), default="soft")
    diagnostic = commands.add_parser("diagnose", help="Measure channel and frequency-band errors from cached latents")
    diagnostic.add_argument("run", type=Path)
    for name in ("evaluate", "sweep"):
        p = commands.add_parser(name, help="Real SD image evaluation" if name == "evaluate" else "Compare a small candidate matrix")
        p.add_argument("--checkpoint", default="models/sd15/v1-5-pruned-emaonly.safetensors")
        p.add_argument("--clip-dir", default="models/sd15/clip")
        p.add_argument("--config", default="configs/stable-diffusion/ldm.yaml")
        p.add_argument("--prompts", default="text_prompt_dataset/chatgpt_dataset.txt")
        p.add_argument("--split", choices=("dev", "validation", "final"), default="dev")
        p.add_argument("--count", type=int)
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--device", default="auto")
        p.add_argument("--precision", choices=("full", "autocast"), default="full")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--attention", choices=("original", "sdpa"), default="original")
        p.add_argument("--size", type=int, choices=(256, 512), default=512,
                       help="512 reference; 256 optional CPU experiment (report dimensions explicitly)")
        p.add_argument("--generation-steps", type=int, default=20)
        p.add_argument("--inversion-steps", type=int, default=20)
        p.add_argument("--order", type=int, choices=(1, 2, 3), default=2)
        p.add_argument("--guidance", type=float, default=5)
        p.add_argument("--inversion-guidance", type=float, default=1)
        p.add_argument("--posterior", choices=("sample", "mode"), default="mode")
        p.add_argument("--prompt-mode", choices=("empty", "known"), default="empty")
        p.add_argument("--bits", type=int, choices=(1, 2, 4), default=1)
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--decision", choices=("hard", "soft"), default="soft")
        p.add_argument("--whiten", action="store_true")
        p.add_argument("--payload-bits", type=int)
        p.add_argument("--noise-domain", choices=("unit", "signed"), default="unit")
        p.add_argument("--niqe-color", choices=("y", "gray"), default="y")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--reuse-images-from", type=Path,
                       help="Reuse an existing run's carriers only when encoding and generation settings match")
        if name == "sweep":
            p.add_argument("--variants", nargs="+", choices=("baseline", "deterministic", "repeat2", "repeat4", "multibit2", "multibit2_repeat2"),
                           default=["baseline", "deterministic", "repeat2", "repeat4", "multibit2"])
    return root


def doctor():
    modules = ("torch", "torchvision", "numpy", "scipy", "PIL", "cv2", "omegaconf", "pytorch_lightning", "transformers", "einops", "safetensors")
    result = {"python": sys.executable,
              "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
              "checkpoint_ready": Path("models/sd15/v1-5-pruned-emaonly.safetensors").is_file(),
              "clip_config_ready": Path("models/sd15/clip/config.json").is_file()}
    if result["modules"]["torch"]:
        import torch
        result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
    return result


def simulate(trials, output):
    import numpy as np
    from .codec import CodecConfig, SecretCodec
    from .runner import write_json

    if trials < 1:
        raise ValueError("trials must be positive")
    rows = []
    for bits, repeat in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2)):
        config = CodecConfig(bits=bits, repeat=repeat, whiten=repeat > 1)
        soft = SecretCodec(config)
        hard = SecretCodec(replace(config, decision="hard"))
        for sigma in (0.25, 0.5, 1.0):
            accuracies = {"hard": [], "soft": []}
            for trial in range(trials):
                rng = np.random.RandomState(10000 + trial)
                message = rng.randint(0, 2, soft.payload_size).astype(np.uint8)
                latent = soft.encode(message)
                received = latent + rng.normal(0, sigma, latent.shape)
                for name, codec in (("hard", hard), ("soft", soft)):
                    accuracies[name].append(float(np.mean(codec.decode(received) == message)))
            rows.append({"bits": bits, "repeat": repeat, "payload_bits": soft.payload_size,
                         "latent_noise_sigma": sigma,
                         "hard_accuracy": float(np.mean(accuracies["hard"])),
                         "soft_accuracy": float(np.mean(accuracies["soft"]))})
    report = {"kind": "synthetic_latent_channel_only", "course_score": None,
              "note": "No image generation, VAE, JPEG, median filter or NIQE. Not a course benchmark.",
              "trials": trials, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def evaluate(args):
    from .attacks import AttackConfig
    from .backend import ModelConfig, StableDiffusionBackend
    from .codec import CodecConfig
    from .runner import run_evaluation, write_json

    model = ModelConfig(checkpoint=args.checkpoint, clip_dir=args.clip_dir, config=args.config,
                        device=args.device, precision=args.precision, threads=args.threads,
                        attention=args.attention, image_size=args.size,
                        generation_steps=args.generation_steps, inversion_steps=args.inversion_steps,
                        order=args.order, guidance=args.guidance, inversion_guidance=args.inversion_guidance,
                        posterior=args.posterior, prompt_mode=args.prompt_mode)
    codec = CodecConfig(bits=args.bits, repeat=args.repeat, decision=args.decision,
                        whiten=args.whiten, payload_bits=args.payload_bits,
                        shape=(4, args.size // 8, args.size // 8))
    attacks = AttackConfig(noise_domain=args.noise_domain)
    # Validate prompt split before allocating a multi-GB model.
    from .data import select_prompts
    select_prompts(args.prompts, args.split, args.count, args.seed)
    backend = StableDiffusionBackend(model)
    variants = ["single"] if args.command == "evaluate" else args.variants
    results = {}
    generation_cache = {}
    for variant in variants:
        current_codec, current_model = codec, model
        if variant == "baseline":
            current_codec = replace(codec, bits=1, repeat=1, decision="hard", whiten=False)
            current_model = replace(model, posterior="sample")
        elif variant == "deterministic":
            current_codec = replace(codec, bits=1, repeat=1, decision="hard", whiten=False)
            current_model = replace(model, posterior="mode")
        elif variant in ("repeat2", "repeat4"):
            current_codec = replace(codec, bits=1, repeat=int(variant[-1]), decision="soft", whiten=True)
            current_model = replace(model, posterior="mode")
        elif variant.startswith("multibit2"):
            current_codec = replace(codec, bits=2, repeat=2 if variant.endswith("repeat2") else 1, decision="soft", whiten=True)
            current_model = replace(model, posterior="mode")
        backend.config = current_model
        destination = args.output if variant == "single" else args.output / variant
        encoder = dict(vars(current_codec))
        encoder.pop("decision")
        cache_key = json.dumps(encoder, sort_keys=True)
        source = args.reuse_images_from or generation_cache.get(cache_key)
        results[variant] = run_evaluation(backend, current_codec, attacks, args.prompts, destination,
                                         args.split, args.count, args.seed, args.niqe_color, args.resume, source)
        generation_cache.setdefault(cache_key, destination)
        if args.command == "sweep":
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "comparison.json", results)
    return results if args.command == "sweep" else results["single"]


def main():
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "score":
            from .scoring import score
            result = score(args.bpp, args.accuracies, args.niqe).to_dict()
        elif args.command == "simulate":
            result = simulate(args.trials, args.output)
        elif args.command == "redecode":
            from .runner import redecode
            result = redecode(args.run, args.decision)
        elif args.command == "diagnose":
            from .diagnostics import diagnose
            result = diagnose(args.run)
        else:
            result = evaluate(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
