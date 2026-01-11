"""Real URL dataset loader for URLhaus, Kaggle, Tranco data."""
import csv
import json
import random
from pathlib import Path
from typing import Iterator, Tuple, List
import numpy as np
import torch
from torch.utils.data import IterableDataset


class RealURLDataset(IterableDataset):
    """Stream real URLs from CSV/JSONL files with balanced sampling."""
    
    def __init__(
        self,
        malicious_files: List[Path],
        benign_files: List[Path],
        max_len: int = 200,
        samples_per_epoch: int = 2_000_000,
        vocab_size: int = 128
    ):
        self.mal_files = [Path(f) for f in malicious_files if Path(f).exists()]
        self.ben_files = [Path(f) for f in benign_files if Path(f).exists()]
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_class = samples_per_epoch // 2
        self.vocab_size = vocab_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        mal_iter = self._stream(self.mal_files, add_protocol=True)
        ben_iter = self._stream(self.ben_files, add_protocol=True)
        
        mal_count = ben_count = 0
        mal_label = torch.tensor(1.0, dtype=torch.float32)
        ben_label = torch.tensor(0.0, dtype=torch.float32)
        
        while mal_count < self.samples_per_class or ben_count < self.samples_per_class:
            # Alternate malicious/benign for better mixing
            if mal_count < self.samples_per_class:
                try:
                    url = next(mal_iter)
                    yield self._tokenize(url), mal_label
                    mal_count += 1
                except StopIteration:
                    mal_iter = self._stream(self.mal_files, add_protocol=True)
            
            if ben_count < self.samples_per_class:
                try:
                    url = next(ben_iter)
                    yield self._tokenize(url), ben_label
                    ben_count += 1
                except StopIteration:
                    ben_iter = self._stream(self.ben_files, add_protocol=True)
    
    def _stream(self, files: List[Path], add_protocol: bool = True) -> Iterator[str]:
        """Stream URLs from files, handling CSV/JSONL/TXT formats."""
        files = list(files)
        random.shuffle(files)
        
        for path in files:
            suffix = path.suffix.lower()
            try:
                if suffix == '.csv':
                    yield from self._read_csv(path, add_protocol)
                elif suffix == '.jsonl':
                    yield from self._read_jsonl(path)
                elif suffix == '.txt':
                    yield from self._read_txt(path, add_protocol)
            except Exception:
                continue
    
    def _read_csv(self, path: Path, add_protocol: bool) -> Iterator[str]:
        """Read URLs from CSV (URLhaus, Kaggle, Tranco formats)."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Detect URL column
            url_idx = 0
            if header:
                header_lower = [h.lower() for h in header]
                for i, col in enumerate(header_lower):
                    if 'url' in col or col in ['domain', 'site']:
                        url_idx = i
                        break
                # Tranco format: rank,domain (first col is number)
                if header[0].isdigit() or 'tranco' in path.name.lower():
                    url_idx = 1 if len(header) > 1 else 0
            
            for row in reader:
                if len(row) > url_idx:
                    url = row[url_idx].strip()
                    if url and len(url) > 3 and not url.startswith('#'):
                        # Add protocol if missing (domain lists)
                        if add_protocol and not url.startswith('http'):
                            url = f"https://{url}/"
                        yield url
    
    def _read_jsonl(self, path: Path) -> Iterator[str]:
        """Read URLs from JSONL files."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    url = data.get('url', data.get('text', ''))
                    if url and len(url) > 3:
                        yield url
                except:
                    continue
    
    def _read_txt(self, path: Path, add_protocol: bool) -> Iterator[str]:
        """Read URLs from plain text (one per line)."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                url = line.strip()
                if url and len(url) > 3 and not url.startswith('#'):
                    if add_protocol and not url.startswith('http'):
                        url = f"https://{url}/"
                    yield url
    
    def _tokenize(self, url: str) -> torch.Tensor:
        """Convert URL to tensor of character indices."""
        text_bytes = url[:self.max_len].encode('utf-8', errors='ignore')
        arr = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
        arr = arr % self.vocab_size
        
        # Pad to max_len
        if len(arr) < self.max_len:
            padded = np.zeros(self.max_len, dtype=np.int64)
            padded[:len(arr)] = arr
            arr = padded
        
        return torch.from_numpy(arr[:self.max_len])
