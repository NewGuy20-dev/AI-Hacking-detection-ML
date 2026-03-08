#!/usr/bin/env python3
"""Generate realistic benign 60x8 timeseries sequences for the LSTM."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.timeseries_synthetic import generate_stress_aligned_benign_sequences

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "timeseries_benign.npy"
TARGET_SEQUENCES = 1_000_000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Target: {TARGET_SEQUENCES:,} sequences")
    print(f"Output: {OUTPUT_FILE}")

    if OUTPUT_FILE.exists():
        existing = np.load(OUTPUT_FILE, mmap_mode="r").shape[0]
        if existing >= TARGET_SEQUENCES:
            print(f"Already complete with {existing:,} sequences!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")

    batch_size = 100000
    all_batches = []

    for batch_start in tqdm(range(0, TARGET_SEQUENCES, batch_size), desc="Generating"):
        batch_end = min(batch_start + batch_size, TARGET_SEQUENCES)
        batch = generate_stress_aligned_benign_sequences(batch_end - batch_start)
        all_batches.append(batch)

    sequences = np.concatenate(all_batches, axis=0).astype(np.float32)
    np.save(OUTPUT_FILE, sequences)
    print(f"\nGenerated {len(sequences):,} sequences -> {OUTPUT_FILE}")
    print(f"Shape: {sequences.shape}")
    print(f"dtype: {sequences.dtype}")

if __name__ == "__main__":
    main()
