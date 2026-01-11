#!/usr/bin/env python3
"""Download clean text from Wikipedia using Hugging Face datasets."""

import json
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env.local")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "wikipedia_text.jsonl"
TARGET_PARAGRAPHS = 20_000_000

# Get token from environment
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_PARAGRAPHS:,} paragraphs")
    print(f"Output: {OUTPUT_FILE}")
    
    # Check existing - if complete, skip
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_PARAGRAPHS:
            print(f"Already complete with {existing:,} paragraphs!")
            return
        print(f"Found {existing:,} partial. Restarting fresh (streaming can't resume)...")
    
    from datasets import load_dataset
    
    print("Loading Wikipedia dataset from HuggingFace...")
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.en",
        split="train",
        streaming=True
    )
    
    collected = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for article in tqdm(dataset, total=TARGET_PARAGRAPHS, desc="Extracting"):
            if collected >= TARGET_PARAGRAPHS:
                break
            
            text = article.get('text', '')
            if not text:
                continue
                
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            
            for para in paragraphs[:10]:
                if collected >= TARGET_PARAGRAPHS:
                    break
                f.write(json.dumps({
                    'text': para[:2000],
                    'source': 'wikipedia',
                    'type': 'paragraph'
                }) + '\n')
                collected += 1
                
            if collected % 100000 == 0:
                f.flush()
    
    print(f"\nCollected {collected:,} paragraphs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
