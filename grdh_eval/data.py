"""Disjoint deterministic development / validation / final prompt splits."""

import hashlib
import random
from pathlib import Path

SPLITS = {"dev": (0, 10), "validation": (10, 30), "final": (30, 130)}


def select_prompts(path, split="dev", count=None, seed=2026):
    path = Path(path)
    unique = {}
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = text.strip()
        if text and text not in unique:
            unique[text] = {"source_line": line, "prompt": text}
    pool = list(unique.values())
    random.Random(seed).shuffle(pool)
    start, stop = SPLITS[split]
    count = stop - start if count is None else count
    if count < 1 or count > stop - start:
        raise ValueError("count must fit the selected split (%d images maximum)" % (stop - start))
    if len(pool) < start + count:
        raise ValueError("Not enough unique prompts for this disjoint split")
    selected = pool[start:start + count]
    return selected, hashlib.sha256(path.read_bytes()).hexdigest()
