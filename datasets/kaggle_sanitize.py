#!/usr/bin/env python3
import os
import sys
import unicodedata


def sanitize_name(name):
    """Sanitize a single filename or directory name."""
    # Normalize Unicode
    name = unicodedata.normalize("NFKC", name)
    # Replace forbidden characters
    name = name.replace("&", "and")
    # Strip leading/trailing whitespace
    name = name.strip()
    # Collapse multiple spaces
    name = " ".join(name.split())
    return name


def sanitize_directory(root_path):
    """Recursively sanitize all paths in directory (bottom-up)."""
    changes = []
    
    # Collect all paths first to avoid stale references
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for filename in filenames:
            all_paths.append((os.path.join(dirpath, filename), False))
        for dirname in dirnames:
            all_paths.append((os.path.join(dirpath, dirname), True))
    
    # Process paths
    for old_path, is_dir in all_paths:
        if not os.path.exists(old_path):
            continue
            
        name = os.path.basename(old_path)
        new_name = sanitize_name(name)
        
        if new_name != name:
            new_path = os.path.join(os.path.dirname(old_path), new_name)
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                changes.append((old_path, new_path))
    
    return changes


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    if not os.path.isdir(path):
        print(f"Error: {path} is not a directory")
        sys.exit(1)
    
    changes = sanitize_directory(path)
    
    if not changes:
        print("✓ No changes needed - all paths are Kaggle-compatible")
    else:
        print(f"⚠ Fixed {len(changes)} path(s):\n")
        for old, new in changes:
            print(f"  {old}")
            print(f"  → {new}\n")


if __name__ == "__main__":
    main()
