"""Lazy-loaded Stable Diffusion 1.5 adapter, retaining original architectures."""

import gc
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.util import instantiate_from_config

from .attacks import to_uint8
from .codec import SecretCodec


class CheckpointCLIPEmbedder(FrozenCLIPEmbedder):
    """Same text architecture; all text weights come from the SD checkpoint."""

    def __init__(self, version, device="cpu", max_length=77, freeze=True, layer="last", layer_idx=None):
        torch.nn.Module.__init__(self)
        self.tokenizer = CLIPTokenizer.from_pretrained(version, local_files_only=True)
        self.transformer = CLIPTextModel(CLIPTextConfig.from_pretrained(version, local_files_only=True))
        self.device, self.max_length = device, max_length
        self.layer, self.layer_idx = layer, layer_idx
        if layer != "last":
            raise ValueError("This adapter supports the baseline last-layer CLIP condition")
        if freeze:
            self.freeze()


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str = "models/sd15/v1-5-pruned-emaonly.safetensors"
    clip_dir: str = "models/sd15/clip"
    config: str = "configs/stable-diffusion/ldm.yaml"
    device: str = "auto"
    precision: str = "full"
    generation_steps: int = 20
    inversion_steps: int = 20
    order: int = 2
    guidance: float = 5.0
    inversion_guidance: float = 1.0
    posterior: str = "mode"
    prompt_mode: str = "empty"
    threads: int = 4
    attention: str = "original"
    image_size: int = 512

    def __post_init__(self):
        if self.order not in (1, 2, 3) or min(self.generation_steps, self.inversion_steps) < self.order:
            raise ValueError("Sampling steps must be at least the solver order (1..3)")
        if self.posterior not in ("mode", "sample") or self.prompt_mode not in ("empty", "known"):
            raise ValueError("Invalid posterior or prompt mode")
        if self.precision not in ("full", "autocast") or self.threads < 1:
            raise ValueError("Invalid precision or thread count")
        if not np.isfinite([self.guidance, self.inversion_guidance]).all():
            raise ValueError("Guidance values must be finite")
        if self.attention not in ("original", "sdpa"):
            raise ValueError("attention must be original or sdpa")
        if self.image_size != 512:
            raise ValueError("Evaluation uses 512x512 images")


class StableDiffusionBackend:
    def __init__(self, config):
        self.config = config
        checkpoint = Path(config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Missing SD checkpoint. Run scripts/prepare_eval_models.py first: %s" % checkpoint)
        if not Path(config.clip_dir, "config.json").is_file():
            raise FileNotFoundError("Missing local CLIP config/tokenizer: %s" % config.clip_dir)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device if config.device == "auto" else config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu")
        if self.device.type != "cuda" and config.precision == "autocast":
            raise ValueError("Use full precision for the CPU reference path")
        torch.set_num_threads(config.threads)
        architecture = OmegaConf.load(config.config)
        architecture.model.params.cond_stage_config.target = "grdh_eval.backend.CheckpointCLIPEmbedder"
        architecture.model.params.cond_stage_config.params.version = str(Path(config.clip_dir).resolve())
        architecture.model.params.cond_stage_config.params.device = str(self.device)
        print("Creating SD 1.5 model on CPU", flush=True)
        self.model = instantiate_from_config(architecture.model).eval()
        print("Loading checkpoint", checkpoint, flush=True)
        if checkpoint.suffix == ".safetensors":
            from safetensors import safe_open
            # Stream tensor copies, avoiding an extra full checkpoint dictionary.
            with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
                parameters = self.model.state_dict()
                missing = set(parameters) - set(weights.keys())
                essential = [k for k in missing if k.startswith(("model.diffusion_model.", "first_stage_model.", "cond_stage_model.transformer."))]
                if essential:
                    raise ValueError("Checkpoint is incompatible; missing model weights: %s" % essential[:10])
                with torch.no_grad():
                    for key in weights.keys():
                        if key in parameters:
                            value = weights.get_tensor(key)
                            if value.shape != parameters[key].shape:
                                raise ValueError("Checkpoint shape mismatch: %s" % key)
                            parameters[key].copy_(value)
                del parameters
        else:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if any(k.startswith(("model.diffusion_model.", "first_stage_model.", "cond_stage_model.transformer.")) for k in missing):
                raise ValueError("Checkpoint is missing required network parameters")
            del state
        gc.collect()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        if config.attention == "sdpa":
            from .attention import enable_sdpa
            print("Using PyTorch SDPA in %d attention modules" % enable_sdpa(self.model), flush=True)
        self.sampler = DPMSolverSampler(self.model)
        self._conditions = {}
        self._calls = 0
        self._phase = "idle"
        self._phase_started = time.perf_counter()
        self.model.model.diffusion_model.register_forward_hook(self._progress)
        print("Model ready on", self.device, flush=True)

    def _progress(self, module, inputs, output):
        self._calls += 1
        if self._calls == 1 or self._calls % 5 == 0:
            print("  %s: U-Net call %d, elapsed %.1fs" % (
                self._phase, self._calls, time.perf_counter() - self._phase_started), flush=True)

    def _start_phase(self, phase):
        self._phase, self._calls = phase, 0
        self._phase_started = time.perf_counter()

    def scope(self):
        return torch.autocast("cuda") if self.config.precision == "autocast" else nullcontext()

    def conditioning(self, prompt, scale):
        # Cache a bounded set; do not retain all prompt tensors on the GPU.
        if prompt not in self._conditions:
            if len(self._conditions) >= 4:
                self._conditions.clear()
            self._conditions[prompt] = self.model.get_learned_conditioning([prompt])
        c = self._conditions[prompt]
        if scale == 1:
            return c, None
        if "" not in self._conditions:
            self._conditions[""] = self.model.get_learned_conditioning([""])
        return c, self._conditions[""]

    @torch.no_grad()
    def generate(self, latent, prompt):
        self._start_phase("generation")
        expected = self.config.image_size // 8
        if latent.shape != (4, expected, expected):
            raise ValueError("Latent shape does not match the configured image size")
        with self.scope():
            c, uc = self.conditioning(prompt, self.config.guidance)
            start = torch.from_numpy(np.asarray(latent, dtype=np.float32)).unsqueeze(0).to(self.device)
            generated, _ = self.sampler.sample(
                steps=self.config.generation_steps, batch_size=1, shape=latent.shape,
                x_T=start, conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=self.config.guidance, order=self.config.order,
                DPMencode=False, verbose=False)
            image = self.model.decode_first_stage(generated)[0].float().cpu().permute(1, 2, 0).numpy()
        return to_uint8(image / 2 + 0.5)

    @torch.no_grad()
    def invert(self, image_unit, prompt=None, seed=0):
        self._start_phase("inversion")
        if self.config.prompt_mode == "known" and prompt is None:
            raise ValueError("The known-prompt diagnostic requires its shared prompt")
        condition = prompt if self.config.prompt_mode == "known" else ""
        image = np.asarray(image_unit, dtype=np.float32)
        if image.shape != (self.config.image_size, self.config.image_size, 3) or not np.isfinite(image).all():
            raise ValueError("Received RGB image dimensions must match the shared image_size")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Received image must be in [0,1]")
        # Sampling posterior is an explicit baseline ablation, with isolated RNG.
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), self.scope():
            torch.manual_seed(seed)
            x = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
            posterior = self.model.encode_first_stage(x)
            latent = posterior.mode() if self.config.posterior == "mode" else posterior.sample()
            latent = latent * self.model.scale_factor
            c, uc = self.conditioning(condition, self.config.inversion_guidance)
            recovered, _ = self.sampler.sample(
                steps=self.config.inversion_steps, batch_size=1, shape=tuple(latent.shape[1:]),
                x_T=latent, conditioning=c, unconditional_conditioning=uc,
                unconditional_guidance_scale=self.config.inversion_guidance,
                order=self.config.order, DPMencode=True, verbose=False)
        return recovered[0].float().cpu().numpy()


class StegoMethod:
    """Fixed-length bit interface. Only known-prompt diagnostics need a prompt."""

    def __init__(self, backend, codec_config):
        self.backend = backend
        self.codec = SecretCodec(codec_config)

    def embed(self, bits, prompt):
        return self.backend.generate(self.codec.encode(bits), prompt)

    def extract(self, image_unit, prompt=None, seed=0):
        image = np.asarray(image_unit)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        return self.codec.decode(self.backend.invert(image, prompt=prompt, seed=seed))
