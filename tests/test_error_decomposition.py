import unittest
from pathlib import Path

import numpy as np

from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.data import select_prompts
from scripts.run_error_decomposition import aggregate, bit_transition, noise_metrics, observe_generation


class ErrorDecompositionTests(unittest.TestCase):
    def test_transitions_count_corrections_and_new_errors_separately(self):
        result = bit_transition([0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 0, 0, 1, 1])
        self.assertEqual(result, {"fixed": 2, "broken": 1, "net_new_errors": -1})

    def test_confidence_bins_account_for_every_bit_and_error(self):
        codec = SecretCodec(CodecConfig(shape=(1, 4, 4)))
        truth = np.random.RandomState(2).randint(0, 2, codec.payload_size).astype(np.uint8)
        original = codec.encode(truth)
        recovered = original + np.random.RandomState(3).normal(0, 2, original.shape)
        stats, prediction = noise_metrics(codec, recovered, original, truth)
        self.assertEqual(sum(x["count"] for x in stats["confidence_bins"]), len(truth))
        self.assertEqual(sum(x["errors"] for x in stats["confidence_bins"]), int(np.sum(prediction != truth)))
        summary = aggregate([{"stages": {"test": stats}, "transitions": {}, "seconds": 1.0}] * 2)
        self.assertEqual(summary["stages"]["test"]["bit_errors"], 2 * stats["bit_errors"])

    def test_observation_restores_method_after_generation_failure(self):
        class Model:
            def decode_first_stage(self, x):
                return x
        class Backend:
            model = Model()
            def generate(self, noise, prompt):
                raise RuntimeError("failed")
        backend = Backend()
        before = backend.model.decode_first_stage
        with self.assertRaises(RuntimeError):
            observe_generation(backend, None, "")
        self.assertEqual(backend.model.decode_first_stage, before)

    def test_new_holdouts_are_disjoint_from_previous_experiments(self):
        path = Path(__file__).resolve().parents[1] / "text_prompt_dataset/chatgpt_dataset.txt"
        names = ("dev", "validation", "final", "ecc_validation", "ecc_final")
        groups = [{x["prompt"] for x in select_prompts(path, split, seed=2026)[0]} for split in names]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertFalse(left & right)
        self.assertEqual([len(x) for x in groups], [10, 20, 100, 20, 100])


if __name__ == "__main__":
    unittest.main()
