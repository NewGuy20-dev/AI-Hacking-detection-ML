"""Shared utilities for synthetic data generators."""
import json
import time
import sys
from pathlib import Path
from typing import Iterator, Any

class ProgressTracker:
    """Track and display generation progress."""
    def __init__(self, total: int, desc: str = "Generating"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        pct = 100 * self.current / self.total
        sys.stdout.write(f"\r{self.desc}: {self.current:,}/{self.total:,} ({pct:.1f}%) | {rate:.0f}/s | ETA: {eta/60:.1f}m")
        sys.stdout.flush()
    
    def close(self):
        print()

class RatioSampler:
    """Generate samples maintaining benign:malicious ratio."""
    def __init__(self, total: int, benign_ratio: float = 2/3):
        self.total = total
        self.n_benign = int(total * benign_ratio)
        self.n_malicious = total - self.n_benign
    
    def get_counts(self) -> tuple[int, int]:
        return self.n_benign, self.n_malicious

def append_to_jsonl(path: Path, samples: Iterator[dict], batch_size: int = 10000):
    """Append samples to JSONL file in batches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        batch = []
        for sample in samples:
            batch.append(json.dumps(sample, ensure_ascii=False))
            if len(batch) >= batch_size:
                f.write('\n'.join(batch) + '\n')
                batch = []
        if batch:
            f.write('\n'.join(batch) + '\n')

def write_jsonl(path: Path, samples: list[dict]):
    """Write samples to new JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
