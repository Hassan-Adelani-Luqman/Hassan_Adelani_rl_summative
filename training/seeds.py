"""Seed lists shared by training, evaluation, baselines and figures.

Training episodes draw their reset seeds from ``TRAIN_SEED_POOL``. Every model and
baseline is evaluated on the same ``HELD_OUT_SEEDS``, which never appear in training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

TRAIN_SEED_POOL = range(0, 100_000)
TRAIN_EVAL_SEEDS: tuple[int, ...] = tuple(range(0, 50))  # 50 seeds the agents trained on
HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(1_000_000, 1_000_050))  # 50 unseen seeds

# name -> (seeds, scenario); used for the generalisation figure
EVAL_SETS: dict[str, tuple[tuple[int, ...], str]] = {
    "train": (TRAIN_EVAL_SEEDS, "default"),
    "held_out": (HELD_OUT_SEEDS, "default"),
    "heavy_rain": (HELD_OUT_SEEDS, "heavy_rain"),
    "shuffled_hotspots": (HELD_OUT_SEEDS, "shuffled_hotspots"),
}


def seed_hash(seeds: Iterable[int]) -> str:
    """Short fingerprint stored with every evaluation so runs can be checked for comparability."""
    return hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()[:12]
