"""Terminated K=7 convolutional code with optional periodic puncturing.

Generators (171, 133) octal; six zero tail bits are overhead, not payload.
Soft Viterbi takes signed bit evidence, positive for one. No message is used
by the decoder. This module has no dependencies beyond NumPy.
"""

import numpy as np


PATTERNS = {
    "conv12": np.array([[1, 1]], dtype=bool),
    "conv23": np.array([[1, 1], [1, 0]], dtype=bool),
    "conv34": np.array([[1, 1], [1, 0], [0, 1]], dtype=bool),
}


class ConvolutionalCode:
    def __init__(self, kind, channel_bits):
        self.pattern = PATTERNS[kind]
        period = len(self.pattern)
        self.max_payload = (channel_bits // int(self.pattern.sum())) * period - 6
        if self.max_payload < 1:
            raise ValueError("Channel is too short for a terminated convolutional code")
        registers = np.arange(128)
        self.outputs = np.array([[bin(int(r & g)).count("1") % 2 for g in (0o171, 0o133)]
                                 for r in registers], dtype=np.uint8)
        states = np.arange(64)
        self.predecessors = np.stack([states >> 1, (states >> 1) | 32])
        registers = (self.predecessors << 1) | (states & 1)
        self.signs = self.outputs[registers].astype(np.float64) * 2 - 1

    def mask(self, payload_size):
        return np.tile(self.pattern, ((payload_size + 6 + len(self.pattern) - 1) // len(self.pattern), 1))[:payload_size + 6]

    def encode(self, message):
        if not 1 <= len(message) <= self.max_payload:
            raise ValueError("Message length exceeds convolutional coding capacity")
        terminated = np.concatenate([message, np.zeros(6, dtype=np.uint8)])
        output = np.empty((len(terminated), 2), dtype=np.uint8)
        state = 0
        for i, bit in enumerate(terminated):
            register = (state << 1) | int(bit)
            output[i] = self.outputs[register]
            state = register & 63
        return output[self.mask(len(message))]

    def decode(self, evidence, payload_size):
        mask = self.mask(payload_size)
        evidence = np.asarray(evidence, dtype=np.float64)
        if evidence.size < mask.sum() or not np.isfinite(evidence).all():
            raise ValueError("Incomplete or invalid convolutional evidence")
        received = np.zeros(mask.shape)
        received[mask] = evidence[:int(mask.sum())]
        metrics = np.full(64, -np.inf)
        metrics[0] = 0  # Encoder starts in the known all-zero state.
        decisions = np.empty((len(received), 64), dtype=np.uint8)
        for i, pair in enumerate(received):
            candidates = metrics[self.predecessors] + self.signs[:, :, 0] * pair[0] + self.signs[:, :, 1] * pair[1]
            decisions[i] = candidates[1] > candidates[0]
            metrics = np.maximum(candidates[0], candidates[1])
            metrics -= metrics.max()
        # Six tail zeros force the terminal state to zero.
        state = 0
        decoded = np.empty(len(received), dtype=np.uint8)
        for i in range(len(received) - 1, -1, -1):
            decoded[i] = state & 1
            state = (state >> 1) | (int(decisions[i, state]) << 5)
        return decoded[:payload_size]


class HammingCode:
    """Systematic (15,11) code, exact bit-posterior decoding over 2048 words.

The soft metric is four times the binary mapping's signed evidence. A fixed
decoder gain can calibrate that scale on development images. Summing word
probabilities minimizes bit errors rather than whole-block errors under this
channel model. Only original data bits count towards capacity.
"""

    def __init__(self, channel_bits):
        self.max_payload = (channel_bits // 15) * 11
        if self.max_payload < 11:
            raise ValueError("Channel is too short for Hamming coding")
        self.positions = np.array([i for i in range(1, 16) if i not in (1, 2, 4, 8)])
        indices = np.arange(2048)
        self.data = ((indices[:, None] >> np.arange(10, -1, -1)) & 1).astype(np.uint8)
        self.words = np.zeros((2048, 15), dtype=np.uint8)
        self.words[:, self.positions - 1] = self.data
        for parity in (1, 2, 4, 8):
            positions = [i - 1 for i in self.positions if i & parity]
            self.words[:, parity - 1] = self.words[:, positions].sum(axis=1) % 2
        self.signs = self.words.astype(float) * 2 - 1

    def encode(self, message):
        if not 1 <= len(message) <= self.max_payload:
            raise ValueError("Message exceeds Hamming coding capacity")
        padded = np.pad(message, (0, (-len(message)) % 11))
        indices = padded.reshape(-1, 11).dot(1 << np.arange(10, -1, -1))
        return self.words[indices].reshape(-1)

    def decode(self, evidence, payload_size):
        blocks = (payload_size + 10) // 11
        evidence = np.asarray(evidence, dtype=float)
        if evidence.size < blocks * 15 or not np.isfinite(evidence).all():
            raise ValueError("Incomplete or invalid Hamming evidence")
        received = evidence[:blocks * 15].reshape(-1, 15)
        result = []
        # Bounded temporary memory, also for larger latent dimensions.
        for start in range(0, blocks, 128):
            likelihood = 2 * received[start:start + 128].dot(self.signs.T)
            # Unused bits in the last data block are known zero padding.
            if start + 128 >= blocks and payload_size % 11:
                invalid = self.data[:, payload_size % 11:].any(axis=1)
                likelihood[-1, invalid] = -np.inf
            likelihood -= likelihood.max(axis=1, keepdims=True)
            probability = np.exp(likelihood)
            posterior = probability.dot(self.data.astype(float))
            result.append((posterior > .5 * probability.sum(axis=1, keepdims=True)).astype(np.uint8))
        return np.concatenate(result).reshape(-1)[:payload_size]
