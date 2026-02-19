"""Hard-example feedback loop for targeted model improvement."""

from .failure_ingest import FailureRecord, ingest_failures
from .hard_example_generator import HardExampleGenerator
from .replay_buffer import build_replay_dataset, load_baseline_samples, load_previous_hard_examples
from .gating import GatingThresholds, evaluate_gates

__all__ = [
    "FailureRecord",
    "ingest_failures",
    "HardExampleGenerator",
    "build_replay_dataset",
    "load_baseline_samples",
    "load_previous_hard_examples",
    "GatingThresholds",
    "evaluate_gates",
]
