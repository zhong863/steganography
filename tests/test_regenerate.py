import unittest

import numpy as np

from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.regenerate import candidate_pair, refine_bits, symbols_to_latent


class RegenerationTests(unittest.TestCase):
    def setUp(self):
        self.codec = SecretCodec(CodecConfig(shape=(1, 2, 2)))

    def test_candidate_pair_changes_only_one_symbol(self):
        values = np.array([[[.49, .1], [.8, 1.1]]])
        for anchor in ("continuous", "hard"):
            pair = candidate_pair(self.codec, values, 0, anchor)
            zero, one = [self.codec.recover_symbols(x) for x in pair]
            expected = values if anchor == "continuous" else (values > .5).astype(float)
            np.testing.assert_allclose(zero.reshape(-1)[1:], expected.reshape(-1)[1:], atol=1e-7)
            np.testing.assert_allclose(one.reshape(-1)[1:], expected.reshape(-1)[1:], atol=1e-7)
            self.assertAlmostEqual(zero.flat[0], 0, places=6)
            self.assertAlmostEqual(one.flat[0], 1, places=6)

    def test_receiver_corrects_selected_bit_using_images_only(self):
        codec = self.codec

        class FakeGenerator:
            def __init__(self):
                self.prompts = []

            def generate_many(self, noises, prompt):
                self.prompts.append(prompt)
                return np.stack([np.repeat(np.clip(codec.recover_symbols(z)[0], 0, 1)[..., None], 3, axis=2) for z in noises])

        values = np.array([[[.49, .05], [.95, .01]]])
        truth_image = np.repeat(np.array([[1., 0.], [1., 0.]])[..., None], 3, axis=2)
        backend = FakeGenerator()
        before, predictions, trace = refine_bits(backend, codec, truth_image, symbols_to_latent(codec, values), 1)
        self.assertEqual(before[0], 0)
        for after in predictions.values():
            self.assertEqual(after[0], 1)
            np.testing.assert_array_equal(after[1:], before[1:])
        self.assertEqual(backend.prompts, ["", "", ""])
        self.assertEqual({t["index"] for t in trace}, {0})

    def test_tie_keeps_initial_decision(self):
        class SameImages:
            def generate_many(self, noises, prompt):
                return np.full((2, 2, 2, 3), .5)

        values = np.array([[[.51, .1], [.9, .01]]])
        before, predictions, _ = refine_bits(SameImages(), self.codec, np.full((2, 2, 3), .5), symbols_to_latent(self.codec, values), 1)
        for after in predictions.values():
            np.testing.assert_array_equal(after, before)

    def test_plain_codec_projection_matches_actual_encoder(self):
        message = np.array([0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(symbols_to_latent(self.codec, message.reshape(1, 2, 2)), self.codec.encode(message))

    def test_hard_cache_matches_uncached_pairs_with_fewer_generations(self):
        from grdh_eval.regenerate import candidate_distances
        codec = self.codec
        class Generator:
            def __init__(self):
                self.generated = 0
            def generate_many(self, noises, prompt):
                self.generated += len(noises)
                return np.stack([np.repeat(np.clip(codec.recover_symbols(z)[0], 0, 1)[..., None], 3, axis=2) for z in noises])
        values = np.array([[[.49, .51], [.48, .9]]])
        recovered = symbols_to_latent(codec, values)
        received = np.repeat(np.array([[1., 0.], [1., 1.]])[..., None], 3, axis=2)
        cached = Generator()
        _, _, traces = refine_bits(cached, codec, received, recovered, 3, ["hard"])
        self.assertEqual(cached.generated, 4)
        reference = Generator()
        for trace in traces:
            pair = candidate_pair(codec, codec.recover_symbols(recovered), trace["index"], "hard")
            expected = candidate_distances(reference.generate_many(pair, ""), received)
            for metric in expected:
                np.testing.assert_allclose(trace["distances"][metric], expected[metric], atol=0, rtol=0)
        self.assertEqual(reference.generated, 6)

    def test_grade_counts_corrected_bits_and_separates_search_time(self):
        from scripts.run_regeneration_study import reports
        row = {"index": 0, "bpp": .0625, "niqe": 4, "seconds": 50,
               "variants": {"baseline": {}, "continuous_mse": {}}}
        for attack in ("jpeg", "awgn", "mblur"):
            row["search_seconds_" + attack] = 10
            row["variants"]["baseline"]["acc_" + attack] = .9
            row["variants"]["continuous_mse"]["acc_" + attack] = .9 + 1 / 16384
        output = reports([row], 1, "dev", ["baseline", "continuous_mse"])
        difference = output["continuous_mse"]["score"]["total"] - output["baseline"]["score"]["total"]
        self.assertAlmostEqual(difference, 100 / 16384)
        self.assertEqual(output["baseline"]["baseline_seconds"], 20)
        self.assertEqual(output["continuous_mse"]["all_variants_search_seconds"], 30)


if __name__ == "__main__":
    unittest.main()
