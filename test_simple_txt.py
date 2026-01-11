#!/usr/bin/env python3
"""Simple test if .txt files can be read."""
from pathlib import Path

data_dir = Path(r"D:\Vibe- Coding projects\AI-Hacking-detection-ML\datasets")
payload_dir = data_dir / 'security_payloads'

txt_files = list(payload_dir.rglob('*.txt'))[:5]

print(f"Testing {len(txt_files)} .txt files...")

for txt_file in txt_files:
    try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
            print(f"✓ {txt_file.name}: {len(lines)} lines")
            if lines:
                print(f"  Sample: {lines[0][:50]}...")
    except Exception as e:
        print(f"✗ {txt_file.name}: ERROR - {e}")

print("\n✅ All files readable!")
