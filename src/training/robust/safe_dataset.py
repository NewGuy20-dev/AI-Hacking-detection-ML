"""Fault-tolerant dataset wrapper for robust training."""
import os
import time
import logging
from pathlib import Path
from typing import Any, Optional, Set, Callable
from torch.utils.data import Dataset

from .utils import WorkerIndexTracker


class SafeDataset(Dataset):
    """Wrapper that makes any Dataset fault-tolerant.
    
    Features:
    - Catches exceptions in __getitem__ and returns fallback sample
    - Logs index before loading for debugging hangs
    - Tracks bad indices for later analysis
    - Warns on slow sample loading
    """
    
    def __init__(
        self,
        dataset: Dataset,
        fallback_sample: Optional[Any] = None,
        timeout_warning: float = 5.0,
        max_consecutive_failures: int = 10,
        log_every_n: int = 1000,
        tracking_dir: Optional[str] = None,
        on_bad_sample: Optional[Callable[[int, Exception], None]] = None,
    ):
        """
        Args:
            dataset: The underlying dataset to wrap
            fallback_sample: Sample to return on failure (auto-detected if None)
            timeout_warning: Warn if sample takes longer than this (seconds)
            max_consecutive_failures: Raise error after N consecutive failures
            log_every_n: Log progress every N samples
            tracking_dir: Directory for worker index tracking files
            on_bad_sample: Optional callback(idx, exception) for bad samples
        """
        self.dataset = dataset
        self.timeout_warning = timeout_warning
        self.max_consecutive_failures = max_consecutive_failures
        self.log_every_n = log_every_n
        self.on_bad_sample = on_bad_sample
        
        self.logger = logging.getLogger('SafeDataset')
        self.tracker = WorkerIndexTracker(tracking_dir)
        self.bad_indices: Set[int] = set()
        self._consecutive_failures = 0
        self._samples_loaded = 0
        
        # Initialize fallback sample
        self._fallback = fallback_sample
        if self._fallback is None:
            self._fallback = self._find_valid_sample()
    
    def _find_valid_sample(self, max_attempts: int = 100) -> Any:
        """Find a valid sample to use as fallback."""
        for idx in range(min(max_attempts, len(self.dataset))):
            try:
                sample = self.dataset[idx]
                self.logger.info(f"Using index {idx} as fallback sample")
                return sample
            except Exception as e:
                self.logger.debug(f"Index {idx} failed during fallback search: {e}")
                continue
        raise RuntimeError(
            f"Could not find valid fallback sample in first {max_attempts} indices"
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # Track which index we're loading (for debugging hangs)
        self.tracker.set_loading(idx)
        start_time = time.time()
        
        try:
            # Log progress periodically
            self._samples_loaded += 1
            if self._samples_loaded % self.log_every_n == 0:
                self.logger.debug(f"Loaded {self._samples_loaded} samples")
            
            # Attempt to load sample
            sample = self.dataset[idx]
            
            # Check for slow loading
            elapsed = time.time() - start_time
            if elapsed > self.timeout_warning:
                self.logger.warning(
                    f"Slow sample at idx {idx}: {elapsed:.2f}s"
                )
            
            # Reset consecutive failure counter on success
            self._consecutive_failures = 0
            self.tracker.clear()
            return sample
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.warning(f"[SKIP] bad sample at idx {idx}: {type(e).__name__}: {e}")
            
            # Track bad index
            self.bad_indices.add(idx)
            self._consecutive_failures += 1
            
            # Callback for external handling
            if self.on_bad_sample:
                try:
                    self.on_bad_sample(idx, e)
                except Exception:
                    pass
            
            # Check for too many consecutive failures
            if self._consecutive_failures >= self.max_consecutive_failures:
                self.tracker.clear()
                raise RuntimeError(
                    f"Too many consecutive failures ({self._consecutive_failures}). "
                    f"Last error at idx {idx}: {e}"
                )
            
            self.tracker.clear()
            return self._fallback
    
    def get_bad_indices(self) -> Set[int]:
        """Return set of indices that failed to load."""
        return self.bad_indices.copy()
    
    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        self.bad_indices.clear()
        self._consecutive_failures = 0
        self._samples_loaded = 0


class SkipBadIndicesDataset(Dataset):
    """Dataset wrapper that skips known bad indices entirely.
    
    Use this after identifying bad indices with SafeDataset.
    """
    
    def __init__(self, dataset: Dataset, bad_indices: Set[int]):
        self.dataset = dataset
        self.bad_indices = bad_indices
        
        # Build mapping from new indices to original indices
        self._index_map = [
            i for i in range(len(dataset)) if i not in bad_indices
        ]
        
        self.logger = logging.getLogger('SkipBadIndicesDataset')
        self.logger.info(
            f"Skipping {len(bad_indices)} bad indices, "
            f"{len(self._index_map)} samples remaining"
        )
    
    def __len__(self) -> int:
        return len(self._index_map)
    
    def __getitem__(self, idx: int) -> Any:
        original_idx = self._index_map[idx]
        return self.dataset[original_idx]
