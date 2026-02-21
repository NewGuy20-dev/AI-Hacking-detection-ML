"""Tests for replay-buffer construction."""

from src.feedback_loop.replay_buffer import build_replay_dataset


def test_replay_dataset_balances_classes_and_dedupes():
    baseline = [
        {"model": "payload", "text": "a", "label": 1},
        {"model": "payload", "text": "b", "label": 0},
        {"model": "payload", "text": "b", "label": 0},
        {"model": "payload", "text": "c", "label": 1},
    ]
    previous = [{"model": "payload", "text": "hard1", "label": 1}]
    new = [
        {"model": "payload", "text": "hard2", "label": 0},
        {"model": "payload", "text": "hard3", "label": 1},
    ]

    out = build_replay_dataset(
        model="payload",
        baseline_samples=baseline,
        previous_hard_samples=previous,
        new_hard_samples=new,
        replay_ratio=1.0,
        hard_ratio_cap=0.5,
        seed=1,
    )

    assert out["stats"]["total"] == len(out["samples"])
    positives = sum(1 for s in out["samples"] if s["label"] == 1)
    negatives = sum(1 for s in out["samples"] if s["label"] == 0)
    assert positives == negatives
