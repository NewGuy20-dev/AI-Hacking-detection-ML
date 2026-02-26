"""Tests for meta generator distribution-driven sampling."""
import json

import numpy as np

from src.stress_test.v14.scenarios import MetaGenerator


def test_meta_generator_uses_distribution_file(temp_dir):
    cfg = temp_dir / "score_distributions.json"
    cfg.write_text(
        json.dumps(
            {
                "payload": {"benign": {"mean": 0.1, "std": 0.01}, "attack": {"mean": 0.9, "std": 0.01}},
                "url": {"benign": {"mean": 0.1, "std": 0.01}, "attack": {"mean": 0.9, "std": 0.01}},
                "timeseries": {"benign": {"mean": 0.1, "std": 0.01}, "attack": {"mean": 0.9, "std": 0.01}},
                "network": {"benign": {"mean": 0.1, "std": 0.01}, "attack": {"mean": 0.9, "std": 0.01}},
                "host": {"benign": {"mean": 0.1, "std": 0.01}, "attack": {"mean": 0.9, "std": 0.01}},
            }
        ),
        encoding="utf-8",
    )
    gen = MetaGenerator(seed=42, distributions_path=str(cfg))
    benign = gen._sample_meta_vector(label=0, difficulty="easy")
    attack = gen._sample_meta_vector(label=1, difficulty="easy")
    assert benign.mean() < attack.mean()


def test_meta_generator_forces_disagreement_on_adversarial(temp_dir):
    cfg = temp_dir / "score_distributions.json"
    cfg.write_text(
        json.dumps(
            {
                name: {
                    "benign": {"mean": 0.05, "std": 0.0, "p10": 0.05, "p90": 0.05},
                    "attack": {"mean": 0.95, "std": 0.0, "p10": 0.95, "p90": 0.95},
                }
                for name in ["payload", "url", "timeseries", "network", "host"]
            }
        ),
        encoding="utf-8",
    )
    np.random.seed(7)
    gen = MetaGenerator(seed=7, distributions_path=str(cfg))
    original_noise = gen.DIFFICULTY_NOISE["adversarial"]
    gen.DIFFICULTY_NOISE["adversarial"] = 0.0
    vector = gen._sample_meta_vector(label=1, difficulty="adversarial")
    gen.DIFFICULTY_NOISE["adversarial"] = original_noise
    # With forced disagreement, at least two model scores should be pushed below 0.5.
    assert (vector < 0.5).sum() >= 2
