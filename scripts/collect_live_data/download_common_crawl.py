#!/usr/bin/env python3
"""Download URLs from multiple sources (Tranco, Majestic, Umbrella)."""

import requests
import json
import zipfile
import io
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "common_crawl_urls.jsonl"
TARGET_URLS = 10_000_000  # 10M URLs (realistic)

SOURCES = [
    ("Tranco 1M", "https://tranco-list.eu/top-1m.csv.zip", 1000000),
    ("Majestic 1M", "https://downloads.majestic.com/majestic_million.csv", 1000000),
    ("Umbrella 1M", "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip", 1000000),
]

def download_tranco():
    """Download Tranco top 1M."""
    urls = []
    try:
        print("  Downloading Tranco...")
        resp = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open('top-1m.csv') as f:
                for line in f:
                    parts = line.decode().strip().split(',')
                    if len(parts) >= 2:
                        urls.append(f"https://{parts[1]}/")
        print(f"  ✓ Tranco: {len(urls):,} URLs")
    except Exception as e:
        print(f"  ✗ Tranco failed: {e}")
    return urls

def download_majestic():
    """Download Majestic Million."""
    urls = []
    try:
        print("  Downloading Majestic...")
        resp = requests.get("https://downloads.majestic.com/majestic_million.csv", timeout=60, stream=True)
        for i, line in enumerate(resp.iter_lines()):
            if i == 0:  # Skip header
                continue
            if i > 1000000:
                break
            parts = line.decode().split(',')
            if len(parts) >= 3:
                urls.append(f"https://{parts[2]}/")
        print(f"  ✓ Majestic: {len(urls):,} URLs")
    except Exception as e:
        print(f"  ✗ Majestic failed: {e}")
    return urls

def download_umbrella():
    """Download Cisco Umbrella top 1M."""
    urls = []
    try:
        print("  Downloading Umbrella...")
        resp = requests.get("http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open('top-1m.csv') as f:
                for line in f:
                    parts = line.decode().strip().split(',')
                    if len(parts) >= 2:
                        urls.append(f"https://{parts[1]}/")
        print(f"  ✓ Umbrella: {len(urls):,} URLs")
    except Exception as e:
        print(f"  ✗ Umbrella failed: {e}")
    return urls

def generate_url_variations(base_urls, target):
    """Generate realistic URL variations from base domains."""
    paths = ['', '/about', '/contact', '/products', '/services', '/blog', '/news', 
             '/login', '/signup', '/api', '/docs', '/help', '/faq', '/terms', '/privacy',
             '/search?q=test', '/page/1', '/category/all', '/user/profile', '/settings']
    
    urls = []
    for url in tqdm(base_urls, desc="Generating variations"):
        if len(urls) >= target:
            break
        domain = url.rstrip('/')
        for path in paths:
            urls.append(f"{domain}{path}")
            if len(urls) >= target:
                break
    return urls

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_URLS:,} URLs")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE))
        if existing >= TARGET_URLS:
            print(f"Already complete with {existing:,} URLs!")
            return
        print(f"Found {existing:,} partial. Restarting...")
    
    # Download from multiple sources
    print("\nDownloading URL lists...")
    all_urls = []
    all_urls.extend(download_tranco())
    all_urls.extend(download_majestic())
    all_urls.extend(download_umbrella())
    
    # Deduplicate
    unique_urls = list(set(all_urls))
    print(f"\nUnique base URLs: {len(unique_urls):,}")
    
    # Generate variations to reach target
    if len(unique_urls) < TARGET_URLS:
        print(f"Generating variations to reach {TARGET_URLS:,}...")
        unique_urls = generate_url_variations(unique_urls, TARGET_URLS)
    
    # Write output
    print(f"\nWriting {len(unique_urls):,} URLs...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in tqdm(unique_urls[:TARGET_URLS], desc="Writing"):
            f.write(json.dumps({"text": url, "source": "top_domains", "type": "url"}) + '\n')
    
    print(f"\n✓ Saved {min(len(unique_urls), TARGET_URLS):,} URLs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
