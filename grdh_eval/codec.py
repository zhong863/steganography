"""Real-bit GRDH mapping with optional repetition and soft decoding.

Shared settings fix the payload length, transform keys and image dimensions.
The decoder receives ONLY a recovered latent, never the original message.
RandomState instances are local: global NumPy state is never reset.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import qr


@dataclass(frozen=True)
class CodecConfig:
    bits: int = 1
    repeat: int = 1
    decision: str = "soft"
    whiten: bool = False
    seed_kernel: int = 100
    seed_shuffle: int = 101
    seed_code: int = 102
    shape: Tuple[int, int, int] = (4, 64, 64)
    payload_bits: Optional[int] = None
    ecc: str = "none"
    decode_gain: float = 1.0

    def __post_init__(self):
        if self.bits not in (1, 2, 4):
            raise ValueError("bits must be 1, 2 or 4")
        if self.repeat < 1 or int(self.repeat) != self.repeat:
            raise ValueError("repeat must be a positive integer")
        if self.decision not in ("hard", "soft"):
            raise ValueError("decision must be hard or soft")
        if self.ecc not in ("none", "conv12", "conv23", "conv34", "hamming15"):
            raise ValueError("Unknown error-correcting code")
        if self.ecc == "hamming15" and self.bits != 1:
            raise ValueError("Hamming soft metrics are calibrated for one-bit mapping")
        if self.ecc != "none" and self.repeat != 1:
            raise ValueError("Use either repetition or convolutional coding")
        if not np.isfinite(self.decode_gain) or self.decode_gain <= 0:
            raise ValueError("decode_gain must be finite and positive")
        if len(self.shape) != 3 or any(n <= 0 for n in self.shape):
            raise ValueError("shape must be positive [channels, height, width]")
        if self.shape[-1] != self.shape[-2]:
            raise ValueError("The original double-sided mapping requires square latents")
        for seed in (self.seed_kernel, self.seed_shuffle, self.seed_code):
            if not 0 <= seed < 2 ** 32:
                raise ValueError("Seeds must be uint32 values")
        if self.payload_bits is not None and not 1 <= self.payload_bits <= self.max_payload:
            raise ValueError("payload_bits exceeds the coding capacity")

    @property
    def channel_bits(self):
        return int(np.prod(self.shape)) * self.bits

    @property
    def max_payload(self):
        if self.ecc != "none":
            from .ecc import ConvolutionalCode, HammingCode
            code = HammingCode(self.channel_bits) if self.ecc == "hamming15" else ConvolutionalCode(self.ecc, self.channel_bits)
            return code.max_payload
        return self.channel_bits // self.repeat

    @property
    def payload_size(self):
        return self.max_payload if self.payload_bits is None else self.payload_bits


def validate_bits(bits):
    array = np.asarray(bits)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise ValueError("A one-dimensional array containing only 0 and 1 is required")
    return array.astype(np.uint8)


def bits_to_symbols(bits, width):
    """Interpret each group as a Gray label, then invert Gray to its index."""
    bits = validate_bits(bits)
    if bits.size % width:
        raise ValueError("Bit count must be divisible by symbol width")
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    gray = bits.reshape(-1, width).dot(weights).astype(np.int64)
    symbols = gray.copy()
    shift = gray.copy()
    while np.any(shift):
        shift >>= 1
        symbols ^= shift
    return symbols


def symbols_to_bits(symbols, width):
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    gray = symbols ^ (symbols >> 1)
    shifts = np.arange(width - 1, -1, -1)
    return ((gray[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


class SecretCodec:
    def __init__(self, config=CodecConfig()):
        self.config = config
        self.ecc = None
        if config.ecc != "none":
            from .ecc import ConvolutionalCode, HammingCode
            self.ecc = HammingCode(config.channel_bits) if config.ecc == "hamming15" else ConvolutionalCode(config.ecc, config.channel_bits)
        n = config.shape[-1]
        self.kernel = qr(np.random.RandomState(config.seed_kernel).randn(n, n))[0]
        self.permutation = np.random.RandomState(config.seed_shuffle).permutation(
            int(np.prod(config.shape)))
        self.inverse_permutation = np.argsort(self.permutation)
        rng = np.random.RandomState(config.seed_code)
        self.code_permutation = rng.permutation(config.channel_bits) if config.whiten else np.arange(config.channel_bits)
        self.inverse_code_permutation = np.argsort(self.code_permutation)
        self.mask = rng.randint(0, 2, config.channel_bits).astype(np.uint8) if config.whiten else np.zeros(config.channel_bits, dtype=np.uint8)
        self.mean = (2 ** config.bits - 1) / 2
        self.std = ((2 ** (2 * config.bits) - 1) / 12) ** 0.5
        self.labels = symbols_to_bits(np.arange(2 ** config.bits), config.bits).reshape(-1, config.bits)

    @property
    def payload_size(self):
        return self.config.payload_size

    def encode(self, message):
        message = validate_bits(message)
        if message.size != self.payload_size:
            raise ValueError("Expected exactly %d payload bits; got %d" % (self.payload_size, message.size))
        repeated = self.ecc.encode(message) if self.ecc else np.tile(message, self.config.repeat)
        # Padding is not counted as payload. It never enters the decoder output.
        coded = np.random.RandomState(self.config.seed_code ^ 0xA5A5).randint(
            0, 2, self.config.channel_bits).astype(np.uint8)
        coded[:repeated.size] = repeated
        coded = coded[self.code_permutation] ^ self.mask
        symbols = bits_to_symbols(coded, self.config.bits).reshape(self.config.shape)
        standardized = (symbols - self.mean) / self.std
        latent = self.kernel @ standardized @ self.kernel.T
        return latent.reshape(-1)[self.permutation].reshape(self.config.shape).astype(np.float32)

    def recover_symbols(self, latent):
        latent = np.asarray(latent)
        if latent.shape != self.config.shape or not np.isfinite(latent).all():
            raise ValueError("Recovered latent has an invalid shape or nonfinite values")
        latent = latent.reshape(-1)[self.inverse_permutation].reshape(self.config.shape)
        return (self.kernel.T @ latent @ self.kernel) * self.std + self.mean

    def decode(self, latent):
        values = (self.recover_symbols(latent).reshape(-1) - self.mean) * self.config.decode_gain + self.mean
        if self.config.decision == "hard":
            symbols = np.clip(np.rint(values), 0, 2 ** self.config.bits - 1).astype(np.int64)
            evidence = symbols_to_bits(symbols, self.config.bits).astype(np.float64) * 2 - 1
        else:
            # Max-log bit evidence: squared distance to the closest 0/1 label.
            distances = (values[:, None] - np.arange(2 ** self.config.bits)) ** 2
            columns = []
            for bit in range(self.config.bits):
                d0 = distances[:, self.labels[:, bit] == 0].min(axis=1)
                d1 = distances[:, self.labels[:, bit] == 1].min(axis=1)
                columns.append(d0 - d1)
            evidence = np.stack(columns, axis=1).reshape(-1)
        evidence *= 1 - 2 * self.mask.astype(np.int8)
        evidence = evidence[self.inverse_code_permutation]
        if self.ecc:
            return self.ecc.decode(evidence, self.payload_size)
        evidence = evidence[:self.payload_size * self.config.repeat].reshape(self.config.repeat, self.payload_size)
        return (evidence.sum(axis=0) > 0).astype(np.uint8)
