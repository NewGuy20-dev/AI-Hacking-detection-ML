#!/bin/bash

# Navigate to datasets folder
cd datasets/

# Find and rename files with problematic characters
find . -depth -name "*[';%]*" | while read file; do
    # Get directory and filename
    dir=$(dirname "$file")
    base=$(basename "$file")
    
    # Replace problematic characters with underscores
    newbase=$(echo "$base" | sed 's/[";%()'\''<>]/_/g')
    
    # Rename if different
    if [ "$base" != "$newbase" ]; then
        echo "Renaming: $file -> $dir/$newbase"
        mv "$file" "$dir/$newbase"
    fi
done

echo "Cleanup complete!"
