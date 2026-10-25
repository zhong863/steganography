"""The slide-2 formula. No model or third-party dependency is required."""

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class Score:
    capacity: float
    robustness: float
    quality: float

    @property
    def total(self):
        return self.capacity + self.robustness + self.quality

    def to_dict(self):
        return dict(asdict(self), total=self.total)


def score(bpp: float, accuracies: Sequence[float], mean_niqe: float) -> Score:
    """Accuracies are fractions, in JPEG / AWGN / median order.

    Apply the clipping separately to each attack, after averaging its image
    accuracies. NIQE is averaged across CLEAN stego images before clipping.
    """
    if not math.isfinite(bpp) or bpp < 0:
        raise ValueError("bpp must be finite and nonnegative")
    if len(accuracies) != 3:
        raise ValueError("Exactly three attack accuracies are required")
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in accuracies):
        raise ValueError("Accuracies must be finite fractions in [0, 1]")
    if not math.isfinite(mean_niqe) or mean_niqe < 0:
        raise ValueError("NIQE must be finite and nonnegative")
    return Score(
        30 * min(1.0, math.log1p(1000 * bpp) / math.log(1001)),
        50 / 3 * sum(max(0.0, 2 * a - 1) for a in accuracies),
        20 * min(1.0, max(0.0, (8 - mean_niqe) / 5)),
    )
