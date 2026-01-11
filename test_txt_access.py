#!/usr/bin/env python3
"""Test if workers can read security_payloads .txt files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.streaming_dataset import StreamingDataset
import torch

# Get some .txt files
data_dir = Path(r"D:\Vibe- Coding projects\AI-Hacking-detection-ML\datasets")
payload_dir = data_dir / 'security_payloads'

txt_files = list(payload_dir.rglob('*.txt'))[:10]  # Test with 10 files

print(f"Testing with {len(txt_files)} .txt files...")
print(f"Sample files:")
for f in txt_files[:3]:
    print(f"  - {f.name}")

try:
    # Create dataset
    dataset = StreamingDataset(
        file_paths=txt_files,
        max_len=500,
        vocab_size=256,
        samples_per_epoch=100  # Just test 100 samples
    )
    
    # Try to iterate
    print("\nTesting iteration...")
    count = 0
    for tokens, label in dataset:
        count += 1
        if count == 1:
            print(f"  ✓ First sample: tokens shape={tokens.shape}, label={label.item()}")
        if count >= 10:
            break
    
    print(f"  ✓ Successfully read {count} samples from .txt files")
    print("\n✅ Workers CAN access .txt files!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
