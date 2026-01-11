"""Bloom filter registry for tracking training sample hashes."""
import hashlib
import math
import pickle
from pathlib import Path
from typing import Iterator, Optional
import mmh3  # MurmurHash3 - fast non-cryptographic hash


class BloomFilter:
    """Memory-efficient probabilistic set for 96M+ samples."""
    
    def __init__(self, expected_items: int = 100_000_000, fp_rate: float = 0.001):
        self.size = self._optimal_size(expected_items, fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        self.bit_array = bytearray(math.ceil(self.size / 8))
        self.count = 0
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))
    
    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))
    
    def _get_positions(self, item: str) -> list:
        positions = []
        h1 = mmh3.hash(item, 0) % self.size
        h2 = mmh3.hash(item, 1) % self.size
        for i in range(self.hash_count):
            positions.append((h1 + i * h2) % self.size)
        return positions
    
    def add(self, item: str):
        for pos in self._get_positions(item):
            byte_idx, bit_idx = divmod(pos, 8)
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self.count += 1
    
    def __contains__(self, item: str) -> bool:
        for pos in self._get_positions(item):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True
    
    def __len__(self) -> int:
        return self.count


class HashRegistry:
    """Registry for tracking training sample hashes."""
    
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else Path("models/training_hashes.pkl")
        self.bloom: Optional[BloomFilter] = None
    
    def create(self, expected_items: int = 100_000_000):
        self.bloom = BloomFilter(expected_items)
        return self
    
    def add_sample(self, text: str):
        if self.bloom is None:
            raise ValueError("Registry not initialized. Call create() first.")
        h = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
        self.bloom.add(h)
    
    def add_samples(self, texts: Iterator[str], show_progress: bool = False):
        if self.bloom is None:
            raise ValueError("Registry not initialized. Call create() first.")
        for i, text in enumerate(texts):
            self.add_sample(text)
            if show_progress and i % 1_000_000 == 0 and i > 0:
                print(f"  Hashed {i:,} samples...")
    
    def contains(self, text: str) -> bool:
        if self.bloom is None:
            return False
        h = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
        return h in self.bloom
    
    def verify_no_overlap(self, texts: Iterator[str]) -> tuple:
        """Check texts for overlap with training data. Returns (overlap_count, total)."""
        overlaps = 0
        total = 0
        for text in texts:
            total += 1
            if self.contains(text):
                overlaps += 1
        return overlaps, total
    
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'wb') as f:
            pickle.dump(self.bloom, f)
        print(f"Saved hash registry: {len(self.bloom):,} samples, {len(self.bloom.bit_array) / 1024 / 1024:.1f} MB")
    
    def load(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, 'rb') as f:
            self.bloom = pickle.load(f)
        return True
    
    def __len__(self) -> int:
        return len(self.bloom) if self.bloom else 0


if __name__ == "__main__":
    # Test bloom filter
    bf = BloomFilter(expected_items=1000, fp_rate=0.01)
    bf.add("test1")
    bf.add("test2")
    print(f"test1 in bf: {'test1' in bf}")
    print(f"test3 in bf: {'test3' in bf}")
    print(f"Size: {len(bf.bit_array)} bytes")
