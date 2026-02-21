#!/usr/bin/env python3
"""Sanitize dataset filenames for Kaggle upload."""
import os
import re
from pathlib import Path

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    # Replace invalid chars with underscore
    invalid_chars = r'[;<>`\'"|*?\\]'
    return re.sub(invalid_chars, '_', filename)

def sanitize_directory(directory):
    """Recursively sanitize all filenames in directory."""
    directory = Path(directory)
    renamed_count = 0
    
    for root, dirs, files in os.walk(directory, topdown=False):
        root_path = Path(root)
        
        # Sanitize files
        for filename in files:
            sanitized = sanitize_filename(filename)
            if sanitized != filename:
                old_path = root_path / filename
                new_path = root_path / sanitized
                print(f"Renaming: {filename} -> {sanitized}")
                old_path.rename(new_path)
                renamed_count += 1
        
        # Sanitize directories
        for dirname in dirs:
            sanitized = sanitize_filename(dirname)
            if sanitized != dirname:
                old_path = root_path / dirname
                new_path = root_path / sanitized
                print(f"Renaming dir: {dirname} -> {sanitized}")
                old_path.rename(new_path)
                renamed_count += 1
    
    print(f"\nRenamed {renamed_count} items")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sanitize_dataset_filenames.py <directory>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)
    
    sanitize_directory(target_dir)
