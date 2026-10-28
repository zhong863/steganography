"""Run with python -m grdh_eval --help from the repository root."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="PPT-matched scoring and training-free GRDH experiments")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect runtime and local model assets without loading SD")
    s = commands.add_parser("score", help="Calculate the exact slide-2 formula")
    s.add_argument("--bpp", type=float, required=True)
    s.add_argument("--accuracies", type=float, nargs=3, required=True, metavar=("JPEG", "AWGN", "MEDIAN"))
    s.add_argument("--niqe", type=float, required=True)
    red = commands.add_parser("redecode", help="Re-use saved latents to compare hard and soft decoding")
    red.add_argument("run", type=Path)
    red.add_argument("--decision", choices=("hard", "soft"), default="soft")
    p = commands.add_parser("evaluate", help="Real SD image evaluation at 512x512")
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
    p.add_argument("--generation-steps", type=int, default=20)
    p.add_argument("--inversion-steps", type=int, default=20)
    p.add_argument("--order", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--guidance", type=float, default=5)
    p.add_argument("--inversion-guidance", type=float, default=1)
    p.add_argument("--posterior", choices=("sample", "mode"), default="mode")
    p.add_argument("--prompt-mode", choices=("empty", "known"), default="empty")
    p.add_argument("--receiver-filter", choices=("none", "auto_nlm"), default="none")
    p.add_argument("--bits", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--decision", choices=("hard", "soft"), default="soft")
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--payload-bits", type=int)
    p.add_argument("--ecc", choices=("none", "conv12", "conv23", "conv34", "hamming15"), default="none")
    p.add_argument("--decode-gain", type=float, default=1.0)
    p.add_argument("--noise-domain", choices=("unit", "signed"), default="unit")
    p.add_argument("--niqe-color", choices=("y", "gray"), default="y")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reuse-images-from", type=Path,
                   help="Reuse an existing run's carriers only when encoding and generation settings match")
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


def evaluate(args):
    from .attacks import AttackConfig
    from .backend import ModelConfig, StableDiffusionBackend
    from .codec import CodecConfig
    from .runner import run_evaluation

    model = ModelConfig(checkpoint=args.checkpoint, clip_dir=args.clip_dir, config=args.config,
                        device=args.device, precision=args.precision, threads=args.threads,
                        attention=args.attention,
                        generation_steps=args.generation_steps, inversion_steps=args.inversion_steps,
                        order=args.order, guidance=args.guidance, inversion_guidance=args.inversion_guidance,
                        posterior=args.posterior, prompt_mode=args.prompt_mode,
                        receiver_filter=args.receiver_filter)
    codec = CodecConfig(bits=args.bits, repeat=args.repeat, decision=args.decision,
                        whiten=args.whiten, payload_bits=args.payload_bits,
                        ecc=args.ecc, decode_gain=args.decode_gain)
    attacks = AttackConfig(noise_domain=args.noise_domain)
    # Validate prompt split before allocating a multi-GB model.
    from .data import select_prompts
    select_prompts(args.prompts, args.split, args.count, args.seed)
    backend = StableDiffusionBackend(model)
    return run_evaluation(backend, codec, attacks, args.prompts, args.output,
                          split=args.split, count=args.count, seed=args.seed,
                          niqe_color=args.niqe_color, resume=args.resume,
                          generation_source=args.reuse_images_from)


def main():
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "score":
            from .scoring import score
            result = score(args.bpp, args.accuracies, args.niqe).to_dict()
        elif args.command == "redecode":
            from .runner import redecode
            result = redecode(args.run, args.decision)
        else:
            result = evaluate(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
