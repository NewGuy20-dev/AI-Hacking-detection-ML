#!/usr/bin/env python3
"""Generate 10M URL samples (6.67M benign, 3.33M malicious)."""
import argparse
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler, append_to_jsonl

TLDS = ['com', 'org', 'net', 'io', 'co', 'edu', 'gov', 'info', 'biz', 'xyz']
BENIGN_DOMAINS = ['google', 'amazon', 'microsoft', 'apple', 'github', 'stackoverflow', 
                  'wikipedia', 'reddit', 'youtube', 'linkedin', 'twitter', 'facebook']
MALICIOUS_KEYWORDS = ['login', 'secure', 'account', 'verify', 'update', 'confirm', 
                      'banking', 'paypal', 'signin', 'password', 'credential']

def rand_str(n=8): return ''.join(random.choices(string.ascii_lowercase, k=n))
def rand_hex(n=16): return ''.join(random.choices('0123456789abcdef', k=n))

def gen_benign_url():
    """Generate benign URL."""
    patterns = [
        lambda: f"https://www.{random.choice(BENIGN_DOMAINS)}.com/{rand_str()}/{rand_str()}",
        lambda: f"https://{rand_str()}.{random.choice(TLDS)}/",
        lambda: f"https://docs.{rand_str()}.com/api/v{random.randint(1,3)}/{rand_str()}",
        lambda: f"https://cdn.{rand_str()}.net/assets/{rand_str()}.js",
        lambda: f"https://blog.{rand_str()}.io/{random.randint(2020,2026)}/{rand_str()}/",
        lambda: f"https://shop.{rand_str()}.com/products/{rand_str()}-{rand_str()}",
        lambda: f"https://api.{rand_str()}.com/v1/users/{random.randint(1,99999)}",
    ]
    return random.choice(patterns)()

def gen_malicious_url():
    """Generate malicious URL."""
    patterns = [
        # Typosquatting
        lambda: f"https://www.{random.choice(BENIGN_DOMAINS)[::-1]}.com/{random.choice(MALICIOUS_KEYWORDS)}",
        lambda: f"https://{random.choice(BENIGN_DOMAINS)}-{random.choice(MALICIOUS_KEYWORDS)}.{random.choice(TLDS)}/",
        # Phishing
        lambda: f"https://{random.choice(MALICIOUS_KEYWORDS)}-{rand_str()}.{random.choice(TLDS)}/{random.choice(MALICIOUS_KEYWORDS)}.php",
        lambda: f"http://{rand_str()}.{random.choice(TLDS)}/{random.choice(BENIGN_DOMAINS)}/{random.choice(MALICIOUS_KEYWORDS)}",
        # DGA-like
        lambda: f"http://{rand_hex(16)}.{random.choice(TLDS)}/",
        lambda: f"https://{rand_str(20)}.{random.choice(TLDS)}/{rand_hex(8)}.exe",
        # IP-based
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/{rand_str()}.php",
        # Suspicious paths
        lambda: f"https://{rand_str()}.{random.choice(TLDS)}/wp-admin/{random.choice(MALICIOUS_KEYWORDS)}.php",
        lambda: f"https://{rand_str()}.{random.choice(TLDS)}/cgi-bin/{rand_str()}.cgi?cmd={rand_str()}",
    ]
    return random.choice(patterns)()

def generate_samples(n_benign: int, n_malicious: int, output_dir: Path, batch_size: int = 100000):
    """Generate and save samples."""
    total = n_benign + n_malicious
    tracker = ProgressTracker(total, "URL")
    
    benign_path = output_dir / "url_benign_expansion.jsonl"
    malicious_path = output_dir / "url_malicious_expansion.jsonl"
    
    # Generate benign
    batch = []
    for _ in range(n_benign):
        batch.append({"text": gen_benign_url(), "label": 0})
        if len(batch) >= batch_size:
            append_to_jsonl(benign_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    if batch:
        append_to_jsonl(benign_path, iter(batch))
        tracker.update(len(batch))
    
    # Generate malicious
    batch = []
    for _ in range(n_malicious):
        batch.append({"text": gen_malicious_url(), "label": 1})
        if len(batch) >= batch_size:
            append_to_jsonl(malicious_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    if batch:
        append_to_jsonl(malicious_path, iter(batch))
        tracker.update(len(batch))
    
    tracker.close()
    print(f"Saved: {benign_path} ({n_benign:,} samples)")
    print(f"Saved: {malicious_path} ({n_malicious:,} samples)")

def main():
    parser = argparse.ArgumentParser(description="Generate URL samples")
    parser.add_argument("--total", type=int, default=10_000_000, help="Total samples")
    parser.add_argument("--output", type=Path, default=Path("datasets/url_analysis"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} benign + {n_malicious:,} malicious = {args.total:,} total")
    
    generate_samples(n_benign, n_malicious, args.output)

if __name__ == "__main__":
    main()
