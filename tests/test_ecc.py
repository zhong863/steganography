import unittest

import numpy as np

from grdh_eval.codec import CodecConfig, SecretCodec
from grdh_eval.ecc import ConvolutionalCode, HammingCode


class ErrorCorrectionTests(unittest.TestCase):
    def test_blind_denoising_estimates_noise_and_preserves_clean_constant(self):
        from grdh_eval.preprocess import estimate_noise_sigma, denoise_received
        clean = np.full((128, 128, 3), .5, dtype=np.float32)
        noisy = (clean + np.random.RandomState(7).normal(0, .05, clean.shape)).astype(np.float32)
        before = noisy.copy()
        self.assertAlmostEqual(estimate_noise_sigma(noisy), .05, places=2)
        np.testing.assert_array_equal(denoise_received(clean), clean)
        self.assertLess(np.mean((denoise_received(noisy) - clean)**2), np.mean((noisy - clean)**2))
        np.testing.assert_array_equal(noisy, before)

    def test_hamming_all_single_bit_errors_and_partial_payload(self):
        code = HammingCode(15 * 8)
        rng = np.random.RandomState(189)
        for size in (1, 17, code.max_payload):
            message = rng.randint(0, 2, size).astype(np.uint8)
            encoded = code.encode(message)
            for position in range(15):
                received = encoded.astype(float) * 2 - 1
                received[position::15] *= -1
                np.testing.assert_array_equal(code.decode(received * 3, size), message)
        # Check independent parity equations, including all codewords.
        for parity in (1, 2, 4, 8):
            positions = [i - 1 for i in range(1, 16) if i & parity]
            self.assertTrue(np.all(code.words[:, positions].sum(axis=1) % 2 == 0))

    def test_hamming_real_bit_interface(self):
        codec = SecretCodec(CodecConfig(ecc="hamming15", whiten=True, shape=(2, 8, 8)))
        message = np.random.RandomState(5).randint(0, 2, codec.payload_size).astype(np.uint8)
        np.testing.assert_array_equal(codec.decode(codec.encode(message)), message)

    def test_known_impulse_response(self):
        code = ConvolutionalCode("conv12", 32)
        expected = [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0]
        np.testing.assert_array_equal(code.encode(np.array([1, 0])), expected)

    def test_roundtrip_and_effective_capacity(self):
        rng = np.random.RandomState(71)
        for bits in (1, 2, 4):
            for kind in ("conv12", "conv23", "conv34"):
                config = CodecConfig(bits=bits, ecc=kind, whiten=True, shape=(2, 8, 8))
                codec = SecretCodec(config)
                message = rng.randint(0, 2, codec.payload_size).astype(np.uint8)
                self.assertLess(codec.payload_size, config.channel_bits)
                self.assertLessEqual(len(codec.ecc.encode(message)), config.channel_bits)
                np.testing.assert_array_equal(codec.decode(codec.encode(message)), message)

    def test_soft_viterbi_corrects_channel_errors(self):
        rng = np.random.RandomState(912)
        code = ConvolutionalCode("conv12", 8192)
        message = rng.randint(0, 2, code.max_payload).astype(np.uint8)
        encoded = code.encode(message)
        received = encoded.astype(float) * 2 - 1 + rng.normal(0, .6, len(encoded))
        raw_error = np.mean((received > 0) != encoded)
        decoded_error = np.mean(code.decode(received, len(message)) != message)
        self.assertGreater(raw_error, .03)
        self.assertLess(decoded_error, .001)

    def test_gain_is_decoder_only_and_rejects_invalid_config(self):
        message = np.random.RandomState(9).randint(0, 2, 256)
        original = SecretCodec(CodecConfig(bits=2, shape=(2, 8, 8)))
        scaled = SecretCodec(CodecConfig(bits=2, shape=(2, 8, 8), decode_gain=1.1))
        np.testing.assert_array_equal(original.encode(message), scaled.encode(message))
        for options in ({"decode_gain": 0}, {"decode_gain": float("nan")}, {"ecc": "conv12", "repeat": 2}):
            with self.assertRaises(ValueError):
                CodecConfig(**options)


if __name__ == "__main__":
    unittest.main()
