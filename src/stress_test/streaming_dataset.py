"""Streaming dataset for handling 96M+ samples without loading all in RAM."""
import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from torch.utils.data import IterableDataset, DataLoader


class StreamingPayloadDataset(IterableDataset):
    """Streams samples from JSONL files with on-the-fly balancing."""
    
    def __init__(self, 
                 benign_paths: List[Path],
                 malicious_paths: List[Path],
                 max_length: int = 512,
                 buffer_size: int = 10000,
                 balance_ratio: float = 1.0):
        self.benign_paths = [Path(p) for p in benign_paths]
        self.malicious_paths = [Path(p) for p in malicious_paths]
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.balance_ratio = balance_ratio  # benign:malicious ratio
    
    def _stream_jsonl(self, path: Path) -> Iterator[str]:
        """Stream lines from JSONL file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', data.get('payload', ''))
                    if text:
                        yield text[:self.max_length]
                except:
                    continue
    
    def _stream_txt(self, path: Path) -> Iterator[str]:
        """Stream lines from text file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line[:self.max_length]
    
    def _stream_file(self, path: Path) -> Iterator[str]:
        """Stream from file based on extension."""
        if path.suffix == '.jsonl':
            yield from self._stream_jsonl(path)
        else:
            yield from self._stream_txt(path)
    
    def _stream_paths(self, paths: List[Path]) -> Iterator[str]:
        """Stream from multiple paths."""
        for path in paths:
            if path.is_file():
                yield from self._stream_file(path)
            elif path.is_dir():
                for f in path.rglob('*'):
                    if f.is_file() and f.suffix in ['.txt', '.jsonl']:
                        yield from self._stream_file(f)
    
    def __iter__(self) -> Iterator[Tuple[str, int]]:
        """Yield (text, label) tuples with balanced sampling."""
        benign_buffer = []
        malicious_buffer = []
        
        benign_iter = self._stream_paths(self.benign_paths)
        malicious_iter = self._stream_paths(self.malicious_paths)
        
        benign_exhausted = False
        malicious_exhausted = False
        
        while True:
            # Fill buffers
            while len(benign_buffer) < self.buffer_size and not benign_exhausted:
                try:
                    benign_buffer.append(next(benign_iter))
                except StopIteration:
                    benign_exhausted = True
                    break
            
            while len(malicious_buffer) < self.buffer_size and not malicious_exhausted:
                try:
                    malicious_buffer.append(next(malicious_iter))
                except StopIteration:
                    malicious_exhausted = True
                    break
            
            if not benign_buffer and not malicious_buffer:
                break
            
            # Shuffle buffers
            random.shuffle(benign_buffer)
            random.shuffle(malicious_buffer)
            
            # Yield balanced samples
            while benign_buffer or malicious_buffer:
                # Decide which class to yield based on balance ratio
                yield_benign = random.random() < (self.balance_ratio / (1 + self.balance_ratio))
                
                if yield_benign and benign_buffer:
                    yield (benign_buffer.pop(), 0)
                elif malicious_buffer:
                    yield (malicious_buffer.pop(), 1)
                elif benign_buffer:
                    yield (benign_buffer.pop(), 0)
                else:
                    break


def encode_text(text: str, max_length: int = 512) -> List[int]:
    """Encode text to integers (character-level)."""
    encoded = [ord(c) % 256 for c in text[:max_length]]
    # Pad to max_length
    if len(encoded) < max_length:
        encoded.extend([0] * (max_length - len(encoded)))
    return encoded


def collate_fn(batch: List[Tuple[str, int]]):
    """Collate batch of (text, label) into tensors."""
    import torch
    texts, labels = zip(*batch)
    encoded = [encode_text(t) for t in texts]
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def create_streaming_dataloader(
    benign_paths: List[str],
    malicious_paths: List[str],
    batch_size: int = 256,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for streaming dataset."""
    dataset = StreamingPayloadDataset(
        benign_paths=[Path(p) for p in benign_paths],
        malicious_paths=[Path(p) for p in malicious_paths],
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    # Test streaming
    base = Path("/mnt/d/Vibe- Coding projects/AI-Hacking-detection-ML")
    dataset = StreamingPayloadDataset(
        benign_paths=[base / "datasets/benign_5m.jsonl"],
        malicious_paths=[base / "datasets/security_payloads"],
        buffer_size=1000,
    )
    
    count = 0
    for text, label in dataset:
        count += 1
        if count <= 5:
            print(f"[{label}] {text[:50]}...")
        if count >= 100:
            break
    print(f"Streamed {count} samples")
