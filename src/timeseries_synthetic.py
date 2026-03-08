"""Shared synthetic timeseries generators used by training and stress tests."""

from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np


def generate_stress_aligned_normal_sequence(seq_len: int = 60) -> np.ndarray:
    """Match the stress-test benign baseline before difficulty obfuscation."""
    seq = np.zeros((seq_len, 8), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, seq_len)

    seq[:, 0] = 50 + 20 * np.sin(t) + np.random.normal(0, 3, seq_len)
    seq[:, 1] = seq[:, 0] * np.random.uniform(800, 1200) + np.random.normal(0, 500, seq_len)
    seq[:, 2] = np.random.uniform(20, 80) + np.random.normal(0, 5, seq_len)
    seq[:, 3] = np.clip(np.random.exponential(0.02, seq_len), 0, 0.2)
    seq[:, 4:] = np.random.uniform(10, 100, (seq_len, 4))
    return seq.astype(np.float32)


def generate_stress_aligned_benign_sequences(
    n_samples: int,
    difficulties: Optional[Iterable[str]] = None,
    clip_max: float = 50000.0,
) -> np.ndarray:
    """Generate benign sequences using the same family as the stress generator."""
    if n_samples <= 0:
        return np.zeros((0, 60, 8), dtype=np.float32)

    if difficulties is None:
        difficulties = ("easy", "medium", "hard", "adversarial")

    from src.stress_test.v14.difficulty import DifficultyMixin

    difficulty_mixin = DifficultyMixin()
    difficulty_list = tuple(difficulties)
    sequences = []
    for _ in range(int(n_samples)):
        seq = generate_stress_aligned_normal_sequence()
        difficulty = random.choice(difficulty_list)
        seq = difficulty_mixin.apply_difficulty(seq, difficulty, "timeseries")
        seq = np.clip(seq.astype(np.float32), a_min=0.0, a_max=clip_max)
        sequences.append(seq)
    return np.asarray(sequences, dtype=np.float32)
