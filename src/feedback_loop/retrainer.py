"""Training orchestration for feedback-loop datasets."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


MODEL_SCRIPT_MAP = {
    "payload": "src/training/train_payload.py",
    "url": "src/training/train_url.py",
}

MODEL_ARTIFACTS = {
    "payload": ["payload_cnn.pt", "payload_cnn.pth"],
    "url": ["url_cnn.pt", "url_cnn.pth"],
}


@dataclass
class TrainingResult:
    model: str
    status: str
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    command: List[str]

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "status": self.status,
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "stdout_tail": self.stdout[-2000:],
            "stderr_tail": self.stderr[-2000:],
            "command": self.command,
        }


def snapshot_models(models_dir: str | Path, snapshot_dir: str | Path, models: List[str]) -> Dict[str, List[str]]:
    models_dir = Path(models_dir)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, List[str]] = {}
    for model in models:
        copied[model] = []
        for artifact in MODEL_ARTIFACTS.get(model, []):
            source = models_dir / artifact
            if not source.exists():
                continue
            target = snapshot_dir / artifact
            shutil.copy2(source, target)
            copied[model].append(str(target))
    return copied


def restore_models(models_dir: str | Path, snapshot_dir: str | Path, models: List[str]) -> Dict[str, List[str]]:
    models_dir = Path(models_dir)
    snapshot_dir = Path(snapshot_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    restored: Dict[str, List[str]] = {}
    for model in models:
        restored[model] = []
        for artifact in MODEL_ARTIFACTS.get(model, []):
            source = snapshot_dir / artifact
            if not source.exists():
                continue
            target = models_dir / artifact
            shutil.copy2(source, target)
            restored[model].append(str(target))
    return restored


def run_retraining(
    model: str,
    hard_examples_file: str | Path,
    repo_root: str | Path,
    timeout_seconds: int = 7200,
) -> TrainingResult:
    """Run model-specific training script with hard-example file input."""
    script = MODEL_SCRIPT_MAP.get(model)
    if not script:
        return TrainingResult(
            model=model,
            status="failed",
            returncode=2,
            duration_seconds=0.0,
            stdout="",
            stderr=f"Unsupported model for retraining: {model}",
            command=[],
        )

    root = Path(repo_root)
    command = [
        sys.executable,
        str(root / script),
        "--hard-examples-file",
        str(Path(hard_examples_file)),
    ]

    start = time.time()
    try:
        env = os.environ.copy()
        # Avoid Windows cp1252 console crashes when training scripts print symbols like "✓".
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        result = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        duration = time.time() - start
        status = "completed" if result.returncode == 0 else "failed"
        return TrainingResult(
            model=model,
            status=status,
            returncode=result.returncode,
            duration_seconds=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        return TrainingResult(
            model=model,
            status="timeout",
            returncode=124,
            duration_seconds=duration,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            command=command,
        )
