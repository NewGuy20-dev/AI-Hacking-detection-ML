"""Shared utilities for robust training pipeline."""
import os
import sys
import random
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime

import numpy as np
import torch


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_to_file: bool = True
) -> logging.Logger:
    """Setup multiprocessing-safe logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Format includes PID for multiprocessing debugging
    fmt = logging.Formatter(
        '[%(asctime)s] [PID:%(process)d] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler (append mode is atomic for small writes)
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name}_{datetime.now():%Y%m%d}.log",
            mode='a',
            encoding='utf-8'
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger


def atomic_save(obj: Any, path: Path, save_fn=torch.save) -> None:
    """Save object atomically using temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        os.close(fd)
        save_fn(obj, tmp_path)
        # Atomic rename (works on both POSIX and Windows)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_rng_state() -> Dict[str, Any]:
    """Capture all RNG states for reproducibility."""
    state = {
        'torch': torch.get_rng_state().cpu(),  # Ensure CPU tensor
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore all RNG states."""
    if 'torch' in state:
        torch.set_rng_state(state['torch'].cpu())
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'python' in state:
        random.setstate(state['python'])
    if 'cuda' in state and torch.cuda.is_available():
        cuda_states = [s.cpu() for s in state['cuda']]
        torch.cuda.set_rng_state_all(cuda_states)


class BatchTimer:
    """Track batch loading/processing times with rolling statistics."""
    
    def __init__(self, window_size: int = 50, warning_multiplier: float = 3.0):
        self.times = deque(maxlen=window_size)
        self.warning_multiplier = warning_multiplier
        self.logger = logging.getLogger('BatchTimer')
    
    def record(self, duration: float, batch_idx: int) -> bool:
        """Record batch time, return True if anomalously slow."""
        is_slow = False
        if len(self.times) >= 10:
            avg = sum(self.times) / len(self.times)
            if duration > self.warning_multiplier * avg:
                self.logger.warning(
                    f"Slow batch {batch_idx}: {duration:.2f}s (avg: {avg:.2f}s)"
                )
                is_slow = True
        self.times.append(duration)
        return is_slow
    
    @property
    def average(self) -> float:
        """Get average batch time."""
        return sum(self.times) / len(self.times) if self.times else 0.0


class WorkerIndexTracker:
    """Track which dataset indices workers are loading (file-based for multiprocessing)."""
    
    def __init__(self, tracking_dir: Optional[str] = None):
        self.tracking_dir = Path(tracking_dir or tempfile.gettempdir()) / 'dataloader_tracking'
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
    
    def set_loading(self, idx: int) -> None:
        """Mark index as currently being loaded by this worker."""
        pid = os.getpid()
        path = self.tracking_dir / f"worker_{pid}.idx"
        path.write_text(str(idx))
    
    def clear(self) -> None:
        """Clear tracking for this worker."""
        pid = os.getpid()
        path = self.tracking_dir / f"worker_{pid}.idx"
        if path.exists():
            path.unlink()
    
    def get_all_loading(self) -> Dict[int, int]:
        """Get all currently loading indices {pid: idx}."""
        result = {}
        for path in self.tracking_dir.glob("worker_*.idx"):
            try:
                pid = int(path.stem.split('_')[1])
                idx = int(path.read_text().strip())
                result[pid] = idx
            except (ValueError, IOError):
                continue
        return result
    
    def cleanup(self) -> None:
        """Remove all tracking files."""
        for path in self.tracking_dir.glob("worker_*.idx"):
            try:
                path.unlink()
            except IOError:
                pass
