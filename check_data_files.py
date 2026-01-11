#!/usr/bin/env python3
"""Check what data files will be loaded for training."""
from pathlib import Path

data_dir = Path(r"D:\Vibe- Coding projects\AI-Hacking-detection-ML\datasets")

print("=" * 80)
print("DATA FILES CHECK")
print("=" * 80)

# Malicious files
print("\n📛 MALICIOUS FILES:")
payload_dir = data_dir / 'security_payloads'
if payload_dir.exists():
    txt_files = list(payload_dir.rglob('*.txt'))
    jsonl_files = list(payload_dir.rglob('*.jsonl'))
    print(f"  .txt files: {len(txt_files)}")
    print(f"  .jsonl files: {len(jsonl_files)}")
    print(f"  Total malicious: {len(txt_files) + len(jsonl_files)}")
    
    # Show first 5
    print("\n  Sample files:")
    for f in (txt_files + jsonl_files)[:5]:
        size_mb = f.stat().st_size / (1024**2)
        print(f"    - {f.name} ({size_mb:.2f} MB)")
else:
    print("  ✗ security_payloads directory not found!")

# Benign files
print("\n✅ BENIGN FILES:")
benign_dirs = [
    data_dir / 'benign_60m',
    data_dir / 'curated_benign',
    data_dir / 'live_benign',
]

total_benign = 0
for d in benign_dirs:
    if d.exists():
        jsonl_files = list(d.rglob('*.jsonl'))
        txt_files = list(d.rglob('*.txt'))
        count = len(jsonl_files) + len(txt_files)
        total_benign += count
        print(f"  {d.name}: {count} files ({len(jsonl_files)} .jsonl, {len(txt_files)} .txt)")

# Additional benign
for f in ['benign_5m.jsonl', 'fp_test_500k.jsonl']:
    p = data_dir / f
    if p.exists():
        size_mb = p.stat().st_size / (1024**2)
        print(f"  {f}: {size_mb:.2f} MB")
        total_benign += 1

print(f"\n  Total benign: {total_benign}")

print("\n" + "=" * 80)
print(f"SUMMARY:")
print(f"  Malicious files: {len(txt_files) + len(jsonl_files) if payload_dir.exists() else 0}")
print(f"  Benign files: {total_benign}")
print("=" * 80)
