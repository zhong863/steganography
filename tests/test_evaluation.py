"""Metric boundaries, wire-bit correctness, and reference NIQE validation."""

import math
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PIL import Image

from grdh_eval.attacks import AttackConfig, apply_attack
from grdh_eval.codec import CodecConfig, SecretCodec, bits_to_symbols, symbols_to_bits
from grdh_eval.data import select_prompts
from grdh_eval.quality import niqe_rgb
from grdh_eval.runner import summarize
from grdh_eval.scoring import score


class ScoreTests(unittest.TestCase):
    def test_slide_formula_known_example(self):
        result = score(1 / 16, [.9, .9, .9], 4)
        self.assertAlmostEqual(result.total, 74.0251291520442)
        self.assertAlmostEqual(result.capacity, 18.025129152044187)

    def test_clipping_and_exact_maximum(self):
        self.assertEqual(score(1, [1, 1, 1], 3).total, 100)
        self.assertEqual(score(2, [.5, .5, .5], 8).total, 30)
        self.assertEqual(score(0, [0, 0, 0], 10).total, 0)

    def test_clip_each_attack_before_average(self):
        self.assertAlmostEqual(score(.1, [.4, .8, .8], 4).robustness, 20)

    def test_invalid_scores_are_rejected(self):
        for bpp, accuracies, niqe in [(-1, [1] * 3, 4), (1, [90] * 3, 4),
                                       (1, [1, 1], 4), (1, [1] * 3, float("nan"))]:
            with self.assertRaises(ValueError):
                score(bpp, accuracies, niqe)

    def test_aggregate_niqe_before_quality_clipping(self):
        rows = [dict(bpp=.0625, acc_jpeg=.9, acc_awgn=.9, acc_mblur=.9, niqe=n, seconds=1) for n in (1, 9)]
        result = summarize(rows, 2, "dev")
        self.assertEqual(result["score"]["quality"], 12)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(summarize(rows, 100, "final")["status"], "partial")


class CodecTests(unittest.TestCase):
    def test_all_multibit_gray_labels(self):
        for width in (1, 2, 4):
            labels = symbols_to_bits(np.arange(2 ** width), width)
            np.testing.assert_array_equal(bits_to_symbols(labels, width), np.arange(2 ** width))
        np.testing.assert_array_equal(symbols_to_bits([0, 1, 2, 3], 2), [0, 0, 0, 1, 1, 1, 1, 0])

    def test_roundtrip_with_independent_receiver(self):
        for width in (1, 2, 4):
            for repeat in (1, 2, 4):
                for whiten in (False, True):
                    with self.subTest(width=width, repeat=repeat, whiten=whiten):
                        config = CodecConfig(bits=width, repeat=repeat, whiten=whiten, shape=(2, 8, 8))
                        sender, receiver = SecretCodec(config), SecretCodec(config)
                        message = np.random.RandomState(17).randint(0, 2, sender.payload_size)
                        np.testing.assert_array_equal(receiver.decode(sender.encode(message)), message)

    def test_nonuniform_messages(self):
        config = CodecConfig(repeat=4, whiten=True, shape=(2, 8, 8))
        codec = SecretCodec(config)
        for value in (0, 1):
            message = np.full(codec.payload_size, value, dtype=np.uint8)
            np.testing.assert_array_equal(codec.decode(codec.encode(message)), message)

    def test_padding_does_not_count_as_payload(self):
        codec = SecretCodec(CodecConfig(repeat=3, payload_bits=7, shape=(2, 8, 8)))
        message = np.array([1, 0, 0, 1, 1, 0, 1])
        self.assertEqual(codec.payload_size, 7)
        np.testing.assert_array_equal(codec.decode(codec.encode(message)), message)

    def test_capacity_accounts_for_redundancy(self):
        self.assertEqual(SecretCodec().payload_size, 16384)
        self.assertEqual(SecretCodec(CodecConfig(repeat=2)).payload_size, 8192)
        self.assertEqual(SecretCodec(CodecConfig(bits=2, repeat=2)).payload_size, 16384)
        self.assertEqual(SecretCodec(CodecConfig(bits=2)).payload_size, 32768)

    def test_soft_evidence_can_correct_hard_majority_error(self):
        config = CodecConfig(repeat=3, payload_bits=1, shape=(1, 2, 2))
        soft = SecretCodec(config)
        hard = SecretCodec(CodecConfig(repeat=3, payload_bits=1, shape=(1, 2, 2), decision="hard"))
        # Two weak wrong votes versus one strong correct vote for bit 1.
        estimates = np.array([[[-.1, -.1], [2., 1.]]])
        received = (soft.kernel @ estimates @ soft.kernel.T).reshape(-1)[soft.permutation].reshape(config.shape)
        self.assertEqual(int(hard.decode(received)[0]), 0)
        self.assertEqual(int(soft.decode(received)[0]), 1)

    def test_global_rng_unchanged(self):
        np.random.seed(12)
        np.random.rand(19)
        before = np.random.get_state()
        codec = SecretCodec(CodecConfig(whiten=True))
        message = np.zeros(codec.payload_size, dtype=np.uint8)
        codec.decode(codec.encode(message))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_original_mapping_compatibility(self):
        from scripts.mapping_module import ours_mapping
        config = CodecConfig(bits=2, shape=(2, 8, 8), decision="hard")
        codec = SecretCodec(config)
        message = np.random.RandomState(3).randint(0, 2, codec.payload_size)
        symbols = bits_to_symbols(message, 2).reshape((1,) + config.shape)
        original = ours_mapping(bits=2).encode_secret(symbols, seed_kernel=config.seed_kernel, seed_shuffle=config.seed_shuffle)[0]
        np.testing.assert_allclose(codec.encode(message), original, atol=1e-6)

    def test_invalid_messages(self):
        codec = SecretCodec(CodecConfig(shape=(1, 2, 2)))
        for message in ([0, 1], [0, 0, 0, 2], [[0, 1], [1, 0]]):
            with self.assertRaises(ValueError):
                codec.encode(message)
        with self.assertRaises(ValueError):
            CodecConfig(shape=(4, 8, 4))


class ProtocolTests(unittest.TestCase):
    def test_noise_range_and_reproducibility(self):
        clean = np.full((512, 512, 3), 128, dtype=np.uint8)
        unit = apply_attack(clean, "awgn", AttackConfig(noise_domain="unit"), seed=17)
        signed = apply_attack(clean, "awgn", AttackConfig(noise_domain="signed"), seed=17)
        np.testing.assert_array_equal(unit, apply_attack(clean, "awgn", seed=17))
        self.assertAlmostEqual(float(unit.std()), .05, places=3)
        self.assertAlmostEqual(float(signed.std()), .025, places=3)

    def test_three_attacks_preserve_contract(self):
        clean = np.random.RandomState(8).randint(0, 256, (32, 32, 3)).astype(np.uint8)
        before = clean.copy()
        for name in ("jpeg", "awgn", "mblur"):
            result = apply_attack(clean, name)
            self.assertEqual(result.shape, clean.shape)
            self.assertEqual(result.dtype, np.float32)
            self.assertGreaterEqual(result.min(), 0)
            self.assertLessEqual(result.max(), 1)
        np.testing.assert_array_equal(clean, before)

    def test_disjoint_prompt_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("\n".join("prompt %d" % i for i in range(160)), encoding="utf-8")
            groups = [select_prompts(path, split, seed=77)[0] for split in ("dev", "validation", "final")]
            sets = [set(x["prompt"] for x in group) for group in groups]
            self.assertEqual([len(s) for s in sets], [10, 20, 100])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(groups[0], select_prompts(path, "dev", seed=77)[0])

    def test_basic_sr_niqe_reference(self):
        path = Path(__file__).parent / "fixtures/baboon.png"
        with Image.open(path) as image:
            result = niqe_rgb(np.array(image.convert("RGB")))
        self.assertAlmostEqual(result, 5.7295763, places=4)


class AdapterTests(unittest.TestCase):
    def test_optional_sdpa_matches_original_attention(self):
        import torch
        from ldm.modules.attention import CrossAttention
        from grdh_eval.attention import sdpa_forward

        torch.manual_seed(17)
        module = CrossAttention(query_dim=32, context_dim=24, heads=4, dim_head=8).eval()
        x, context = torch.randn(2, 17, 32), torch.randn(2, 13, 24)
        with torch.no_grad():
            original = module(x, context)
            optimized = sdpa_forward(module, x, context)
        torch.testing.assert_close(optimized, original, rtol=1e-5, atol=1e-6)

    def test_evaluator_isolates_receiver_and_generates_once(self):
        from grdh_eval.backend import ModelConfig
        from grdh_eval.runner import run_evaluation

        class FakeBackend:
            def __init__(self, config):
                self.config, self.generate_calls, self.receiver_prompts = config, 0, []

            def generate(self, latent, prompt):
                self.generate_calls += 1
                self.latent = latent
                return np.full((512, 512, 3), 128, dtype=np.uint8)

            def invert(self, image, prompt=None, seed=0):
                self.receiver_prompts.append(prompt)
                return self.latent

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.safetensors"
            checkpoint.write_bytes(b"identity fixture; not real model weights")
            prompts = root / "prompts.txt"
            prompts.write_text("\n".join("prompt %d" % i for i in range(160)), encoding="utf-8")
            backend = FakeBackend(ModelConfig(checkpoint=str(checkpoint), prompt_mode="empty"))
            destination = root / "result"
            with patch("grdh_eval.runner.niqe_rgb", return_value=4), contextlib.redirect_stdout(io.StringIO()):
                result = run_evaluation(backend, CodecConfig(), AttackConfig(), prompts, destination, count=1)
                resumed = run_evaluation(backend, CodecConfig(), AttackConfig(), prompts, destination, count=1, resume=True)
            self.assertEqual(backend.generate_calls, 1)
            self.assertEqual(backend.receiver_prompts, [None, None, None])
            self.assertEqual(result, resumed)
            self.assertFalse(result["course_sample_count_met"])
            # Posterior changes are decoder-only, so carriers may be reused.
            backend.config = replace(backend.config, posterior="sample")
            with patch("grdh_eval.runner.niqe_rgb", return_value=4), contextlib.redirect_stdout(io.StringIO()):
                reused = run_evaluation(backend, CodecConfig(), AttackConfig(), prompts,
                                        root / "reuse", count=1, generation_source=destination)
            self.assertEqual(backend.generate_calls, 1)
            self.assertEqual(reused["score"], result["score"])
            # Changed redundancy is a changed encoder; re-use would be invalid.
            with self.assertRaises(ValueError):
                run_evaluation(backend, CodecConfig(repeat=2), AttackConfig(), prompts,
                               root / "invalid_cache", count=1, generation_source=destination)
            with self.assertRaises(ValueError):
                run_evaluation(backend, CodecConfig(repeat=2), AttackConfig(), prompts, destination, count=1, resume=True)
            stop_file = root / "STOP"
            stop_file.touch()
            with self.assertRaises(KeyboardInterrupt):
                run_evaluation(backend, CodecConfig(), AttackConfig(), prompts,
                               root / "stopped", count=1, stop_file=stop_file)


class StudyTests(unittest.TestCase):
    def test_validation_selects_winner_before_untouched_final_split(self):
        import json
        from scripts import run_cpu_study as study

        calls = []

        def fake_evaluate(backend, codec, attacks, prompts, destination, **kwargs):
            split = kwargs["split"]
            calls.append((split, codec.bits, codec.repeat, kwargs["count"]))
            scores = {(1, 1): 20, (1, 2): 30, (1, 4): 5, (2, 1): 12, (2, 2): 10}
            value = scores[(codec.bits, codec.repeat)]
            if split == "validation":
                value = 45 if codec.repeat == 1 else 35
            return {"score": {"total": value}, "images": kwargs["count"]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.txt"
            prompts.write_text("\n".join("prompt %d" % i for i in range(160)), encoding="utf-8")
            args = ["run_cpu_study.py", "--output", str(root / "study"), "--prompts", str(prompts),
                    "--dev-count", "1", "--validation-count", "1", "--final-count", "1"]
            with patch("sys.argv", args), patch.object(study, "run_evaluation", side_effect=fake_evaluate), \
                    patch("grdh_eval.backend.StableDiffusionBackend", return_value=object()):
                self.assertEqual(study.main(), 0)
            result = json.loads((root / "study/final_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["winner"], "reference")
            self.assertEqual([x[0] for x in calls], ["dev"] * 5 + ["validation"] * 2 + ["final"])
            self.assertEqual(calls[-1][1:3], (1, 1))


if __name__ == "__main__":
    unittest.main()
