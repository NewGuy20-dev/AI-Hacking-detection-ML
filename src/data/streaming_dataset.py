"""Memory-efficient streaming dataset optimized for Intel i5 12th Gen + RTX 3050."""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Iterator, Optional, Tuple
import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """Streams data from JSONL files without loading all into memory."""
    
    def __init__(self, file_paths: List[Path], max_len: int = 500,
                 vocab_size: int = 256, shuffle_files: bool = True,
                 samples_per_epoch: Optional[int] = None):
        self.file_paths = [Path(p) for p in file_paths if Path(p).exists()]
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle_files = shuffle_files
        self.samples_per_epoch = samples_per_epoch
        
        if not self.file_paths:
            raise ValueError("No valid data files found")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        
        files = list(self.file_paths)
        if self.shuffle_files:
            random.shuffle(files)
        
        # Shard files across workers
        worker_files = files[worker_id::num_workers]
        samples_limit = (self.samples_per_epoch // num_workers) if self.samples_per_epoch else None
        
        count = 0
        for path in worker_files:
            for tokens, label in self._read_file(path):
                yield tokens, label
                count += 1
                if samples_limit and count >= samples_limit:
                    return
    
    def _read_file(self, path: Path) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Read file - handles both JSONL and plain text."""
        suffix = path.suffix.lower()
        if suffix == '.jsonl':
            yield from self._read_jsonl(path)
        else:
            # Plain text file - each line is a sample
            yield from self._read_txt(path, label=1)  # Default label for txt
    
    def _read_jsonl(self, path: Path) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore', buffering=1<<20) as f:
                for line in f:
                    if len(line) < 3:
                        continue
                    try:
                        data = json.loads(line)
                        text = str(data.get('text', data.get('payload', '')))
                        label = float(data.get('label', data.get('is_malicious', 0)))
                        tokens = self._tokenize_numpy(text)
                        yield tokens, torch.tensor(label, dtype=torch.float32)
                    except:
                        continue
        except:
            pass
    
    def _read_txt(self, path: Path, label: int = 1) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Read plain text file - each line is a sample."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 3:
                        tokens = self._tokenize_numpy(line)
                        yield tokens, torch.tensor(float(label), dtype=torch.float32)
        except:
            pass
    
    def _tokenize_numpy(self, text: str) -> torch.Tensor:
        """Vectorized tokenization using numpy."""
        text_bytes = text[:self.max_len].encode('utf-8', errors='ignore')
        arr = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
        arr = arr % self.vocab_size
        # Pad
        if len(arr) < self.max_len:
            arr = np.pad(arr, (0, self.max_len - len(arr)), mode='constant')
        return torch.from_numpy(arr[:self.max_len])


class BalancedStreamingDataset(IterableDataset):
    """Optimized balanced streaming with worker sharding and numpy vectorization."""
    
    def __init__(self, malicious_files: List[Path], benign_files: List[Path],
                 max_len: int = 500, samples_per_epoch: int = 20_000_000, vocab_size: int = 256):
        self.malicious_files = [Path(p) for p in malicious_files if Path(p).exists()]
        self.benign_files = [Path(p) for p in benign_files if Path(p).exists()]
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_class = samples_per_epoch // 2
        self.vocab_size = vocab_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        
        samples_per_worker = self.samples_per_class // num_workers
        
        # Pre-create label tensors (reused)
        mal_label = torch.tensor(1.0, dtype=torch.float32)
        ben_label = torch.tensor(0.0, dtype=torch.float32)
        
        mal_iter = self._stream_sharded(self.malicious_files, worker_id, num_workers)
        ben_iter = self._stream_sharded(self.benign_files, worker_id, num_workers)
        
        mal_count = ben_count = 0
        
        while mal_count < samples_per_worker or ben_count < samples_per_worker:
            if mal_count < samples_per_worker:
                try:
                    tokens = next(mal_iter)
                    yield tokens, mal_label
                    mal_count += 1
                except StopIteration:
                    mal_iter = self._stream_sharded(self.malicious_files, worker_id, num_workers)
            
            if ben_count < samples_per_worker:
                try:
                    tokens = next(ben_iter)
                    yield tokens, ben_label
                    ben_count += 1
                except StopIteration:
                    ben_iter = self._stream_sharded(self.benign_files, worker_id, num_workers)
    
    def _stream_sharded(self, files: List[Path], worker_id: int, num_workers: int) -> Iterator[torch.Tensor]:
        """Stream files with worker sharding."""
        files = list(files)
        random.shuffle(files)
        worker_files = files[worker_id::num_workers] if num_workers > 1 else files
        
        for path in worker_files:
            yield from self._read_file_fast(path)
    
    def _read_file_fast(self, path: Path) -> Iterator[torch.Tensor]:
        """Optimized file reading - handles both JSONL and plain text."""
        suffix = path.suffix.lower()
        is_jsonl = suffix == '.jsonl'
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore', buffering=1<<20) as f:
                batch_lines = []
                for line in f:
                    if len(line) >= 3:
                        batch_lines.append(line)
                    
                    if len(batch_lines) >= 1000:
                        yield from self._process_batch(batch_lines, is_jsonl)
                        batch_lines = []
                
                if batch_lines:
                    yield from self._process_batch(batch_lines, is_jsonl)
        except:
            pass
    
    def _process_batch(self, lines: List[str], is_jsonl: bool = True) -> Iterator[torch.Tensor]:
        """Process a batch of lines."""
        for line in lines:
            if is_jsonl:
                text = self._extract_text_fast(line)
            else:
                # Plain text - line itself is the sample
                text = line.strip()
            if text and len(text) > 3:
                yield self._tokenize_numpy(text)
    
    def _extract_text_fast(self, line: str) -> Optional[str]:
        """Fast text extraction without full JSON parse."""
        try:
            # Try fast path first
            for key in ('"text"', '"payload"'):
                idx = line.find(key)
                if idx != -1:
                    start = line.find(':', idx) + 1
                    # Skip whitespace and opening quote
                    while start < len(line) and line[start] in ' \t"':
                        start += 1
                    # Find closing quote
                    end = line.find('"', start)
                    if end == -1:
                        end = min(line.find(',', start), line.find('}', start))
                        if end == -1:
                            end = len(line)
                    return line[start:end] if end > start else None
            
            # Fallback to JSON
            data = json.loads(line)
            return str(data.get('text', data.get('payload', '')))
        except:
            return None
    
    def _tokenize_numpy(self, text: str) -> torch.Tensor:
        """Vectorized tokenization using numpy - no Python loops."""
        # Encode to bytes and convert to numpy array
        text_bytes = text[:self.max_len].encode('utf-8', errors='ignore')
        arr = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
        
        # Clamp to vocab_size to prevent index out of bounds
        arr = arr % self.vocab_size
        
        # Pad to max_len
        if len(arr) < self.max_len:
            padded = np.zeros(self.max_len, dtype=np.int64)
            padded[:len(arr)] = arr
            arr = padded
        else:
            arr = arr[:self.max_len]
        
        return torch.from_numpy(arr)
