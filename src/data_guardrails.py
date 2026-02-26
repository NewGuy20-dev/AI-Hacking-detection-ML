"""Training data guardrails for prohibited evaluation datasets."""
from pathlib import Path
from typing import Iterable, Union


PathLike = Union[str, Path]

PROHIBITED_TRAINING_PATHS = (
    Path("datasets/holdout_test"),
    Path("datasets/fp_test_500k.jsonl"),
)


def _normalize(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def assert_allowed_training_path(path: PathLike, context: str = "training input") -> None:
    """Raise ValueError when a path points to prohibited training data."""
    resolved = _normalize(path)
    root = _normalize(Path(__file__).resolve().parent.parent)

    for blocked in PROHIBITED_TRAINING_PATHS:
        blocked_abs = _normalize(root / blocked)
        if resolved == blocked_abs:
            raise ValueError(
                f"Blocked {context}: {resolved}. "
                f"Do not train on {blocked.as_posix()}."
            )
        if blocked_abs.is_dir() and blocked_abs in resolved.parents:
            raise ValueError(
                f"Blocked {context}: {resolved}. "
                f"Do not train on any data inside {blocked.as_posix()}/."
            )


def assert_allowed_training_paths(paths: Iterable[PathLike], context: str = "training input") -> None:
    """Validate a list of paths before reading data for training."""
    for path in paths:
        assert_allowed_training_path(path, context=context)
