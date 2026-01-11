#!/usr/bin/env python3
"""Generate realistic benign URLs locally (no network needed)."""

import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "common_crawl_urls.jsonl"
TARGET_URLS = 10_000_000

# Real TLDs and patterns
TLDS = ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'us', 'uk', 'de', 'fr', 'jp', 'cn', 'au', 'ca']
WORDS = ['shop', 'store', 'buy', 'news', 'blog', 'tech', 'web', 'app', 'cloud', 'data', 'info', 
         'home', 'best', 'top', 'my', 'the', 'get', 'go', 'pro', 'plus', 'hub', 'lab', 'dev',
         'code', 'learn', 'edu', 'health', 'food', 'travel', 'music', 'video', 'photo', 'game',
         'sport', 'auto', 'finance', 'bank', 'insurance', 'real', 'estate', 'job', 'career']
PATHS = ['', '/', '/about', '/contact', '/products', '/services', '/blog', '/news', '/help',
         '/login', '/signup', '/api/v1', '/docs', '/faq', '/terms', '/privacy', '/search',
         '/category/all', '/user/profile', '/settings', '/dashboard', '/home', '/index.html']
SUBDOMAINS = ['', 'www.', 'app.', 'api.', 'blog.', 'shop.', 'mail.', 'cdn.', 'm.', 'dev.']

def generate_domain():
    """Generate a realistic domain name."""
    pattern = random.choice([
        lambda: f"{random.choice(WORDS)}{random.choice(WORDS)}",
        lambda: f"{random.choice(WORDS)}-{random.choice(WORDS)}",
        lambda: f"{random.choice(WORDS)}{random.randint(1, 999)}",
        lambda: f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))}",
        lambda: f"{random.choice(WORDS)}{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))}",
    ])
    return f"{pattern()}.{random.choice(TLDS)}"

def generate_url():
    """Generate a realistic benign URL."""
    subdomain = random.choice(SUBDOMAINS)
    domain = generate_domain()
    path = random.choice(PATHS)
    
    # Sometimes add query params
    if random.random() < 0.2:
        params = ['id', 'page', 'q', 'ref', 'source', 'utm_source', 'lang']
        path += f"?{random.choice(params)}={random.randint(1, 1000)}"
    
    protocol = 'https' if random.random() < 0.9 else 'http'
    return f"{protocol}://{subdomain}{domain}{path}"

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
    
    print("Generating benign URLs locally...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(TARGET_URLS), desc="Generating"):
            url = generate_url()
            f.write(json.dumps({"text": url, "source": "generated", "type": "url"}) + '\n')
    
    print(f"\n✓ Generated {TARGET_URLS:,} URLs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
