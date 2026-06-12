"""Deterministic train/val/test split assignment shared by all adapters.

Splitting by a hash of a stable key (instead of `random.shuffle`) means
re-running the pipeline never reshuffles items that were already assigned,
even if the dataset grows later.
"""

from __future__ import annotations

import hashlib


def _hash_fraction(key: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) / 16 ** len(digest)


def assign_split(key: str, ratios: dict[str, float], seed: int = 42) -> str:
    """Deterministically assign `key` to one of `ratios`' buckets.

    `ratios` maps split name -> fraction and should sum to ~1.0, e.g.
    `{"train": 0.7, "val": 0.15, "test": 0.15}`. Order matters: buckets are
    consumed in iteration order.
    """
    fraction = _hash_fraction(key, seed)
    cumulative = 0.0
    for split, ratio in ratios.items():
        cumulative += ratio
        if fraction < cumulative:
            return split
    return next(reversed(ratios))  # fallback for floating-point rounding
