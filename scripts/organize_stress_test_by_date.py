#!/usr/bin/env python3
"""Organize stress test files into date-based folders."""
import re
from pathlib import Path
from collections import defaultdict

def organize_stress_test_files(base_dir: str = "evaluation/stress_test_v14"):
    """Move stress test files into date-based subdirectories."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Directory not found: {base_path}")
        return
    
    # Pattern to extract date from filename
    date_pattern = re.compile(r'_(\d{4}-\d{2}-\d{2})')
    
    # Group files by date
    files_by_date = defaultdict(list)
    
    for file in base_path.iterdir():
        if file.is_file() and file.name not in ['.gitkeep']:
            match = date_pattern.search(file.name)
            if match:
                date_str = match.group(1)
                files_by_date[date_str].append(file)
    
    # Move files into date folders
    for date_str, files in sorted(files_by_date.items()):
        date_folder = base_path / date_str
        date_folder.mkdir(exist_ok=True)
        
        print(f"\n📁 {date_str}/ ({len(files)} files)")
        for file in files:
            dest = date_folder / file.name
            if not dest.exists():
                file.rename(dest)
                print(f"  ✓ {file.name}")
            else:
                print(f"  ⚠ {file.name} (already exists)")
    
    print(f"\n✅ Organization complete!")

if __name__ == '__main__':
    organize_stress_test_files()
