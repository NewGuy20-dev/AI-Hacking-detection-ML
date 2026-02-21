#!/usr/bin/env python3
"""Download DARPA dataset using curl."""
import subprocess
import os
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / 'datasets' / 'darpa'
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Direct download links from MIT LL archive
DARPA_FILES = {
    'tcpdump.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/tcpdump.gz',
    'bsm.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/bsm.gz',
    'root.dump.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/root.dump.gz',
    'usr.dump.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/usr.dump.gz',
    'home.dump.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/home.dump.gz',
    'opt.dump.gz': 'https://archive.ll.mit.edu/ideval/data/1998/training/four_hours/opt.dump.gz',
}

def download_file(url, filename):
    """Download file using curl."""
    filepath = DATASETS_DIR / filename
    print(f"\n📥 Downloading {filename}...")
    cmd = f'curl -L -o "{filepath}" "{url}"'
    try:
        subprocess.run(cmd, shell=True, check=True)
        size = filepath.stat().st_size / (1024*1024)
        print(f"✅ {filename} ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ {filename} - {e}")
        return False

def main():
    print("="*60)
    print("📥 DARPA Dataset Download (curl)")
    print("="*60)
    
    results = {}
    for filename, url in DARPA_FILES.items():
        results[filename] = download_file(url, filename)
    
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    for filename, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {filename}")
    
    print(f"\n📁 Location: {DATASETS_DIR}")
    print(f"📊 Progress: {success}/{total} files")
    print(f"💾 Total size: ~221 MB")

if __name__ == '__main__':
    main()
