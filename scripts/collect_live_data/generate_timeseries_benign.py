#!/usr/bin/env python3
"""Generate realistic benign timeseries sequences for LSTM."""

import numpy as np
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "timeseries_benign.npy"
TARGET_SEQUENCES = 1_000_000

PATTERNS = {
    'steady': {'base': 100, 'var': 10, 'trend': 0},
    'daily': {'base': 100, 'var': 20, 'trend': 0, 'cycle': True},
    'growing': {'base': 80, 'var': 15, 'trend': 0.5},
    'declining': {'base': 120, 'var': 15, 'trend': -0.3},
    'bursty': {'base': 50, 'var': 30, 'trend': 0},
}

def generate_sequence(pattern_name: str, seq_len: int = 60, n_features: int = 4) -> np.ndarray:
    p = PATTERNS[pattern_name]
    seq = np.zeros((seq_len, n_features))
    
    for f in range(n_features):
        signal = np.random.normal(p['base'] + f * 10, p['var'], seq_len)
        signal += np.arange(seq_len) * p['trend']
        if p.get('cycle'):
            signal += np.sin(np.linspace(0, 2 * np.pi, seq_len)) * p['var'] * 0.5
        seq[:, f] = np.maximum(signal, 0)
    
    return seq.astype(np.float32)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_SEQUENCES:,} sequences")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = np.load(OUTPUT_FILE).shape[0]
        if existing >= TARGET_SEQUENCES:
            print(f"Already complete with {existing:,} sequences!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    patterns = list(PATTERNS.keys())
    batch_size = 100000
    all_seqs = []
    
    for batch_start in tqdm(range(0, TARGET_SEQUENCES, batch_size), desc="Generating"):
        batch_end = min(batch_start + batch_size, TARGET_SEQUENCES)
        batch = [generate_sequence(np.random.choice(patterns)) for _ in range(batch_end - batch_start)]
        all_seqs.extend(batch)
    
    np.save(OUTPUT_FILE, np.stack(all_seqs))
    print(f"\nGenerated {len(all_seqs):,} sequences -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
