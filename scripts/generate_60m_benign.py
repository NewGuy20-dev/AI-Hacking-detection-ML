#!/usr/bin/env python3
"""Generate 60M benign samples for training data expansion."""
import json
import hashlib
import argparse
import random
import string
from pathlib import Path
from tqdm import tqdm

# Import generators directly to avoid src/__init__.py issues
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))
from benign_generators import (
    generate_urls, generate_sql, generate_shell, generate_api_calls,
    generate_code_snippets, generate_logs, generate_configs, generate_text
)


class HashRegistry:
    """Track hashes to avoid duplicates."""
    def __init__(self):
        self.seen = set()
    
    def add(self, text: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        if h in self.seen:
            return False
        self.seen.add(h)
        return True


def load_tranco_domains(path: Path, limit: int = 1_000_000) -> list:
    """Load domains from Tranco top-1m list."""
    domains = []
    if not path.exists():
        print(f"Warning: {path} not found, using fallback domains")
        return ["google.com", "facebook.com", "amazon.com", "microsoft.com", 
                "apple.com", "github.com", "stackoverflow.com", "wikipedia.org"]
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.strip().split(',')
            if len(parts) >= 2:
                domains.append(parts[1])
            else:
                domains.append(parts[0])
    return domains


def write_jsonl(output_path: Path, generator, count: int, registry: HashRegistry, desc: str):
    """Write samples to JSONL file with deduplication."""
    written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(total=count, desc=desc)
        for text in generator:
            if registry.add(text):
                f.write(json.dumps({"text": text, "label": 0}) + '\n')
                written += 1
                pbar.update(1)
                if written >= count:
                    break
        pbar.close()
    return written


def main():
    parser = argparse.ArgumentParser(description='Generate 60M benign samples')
    parser.add_argument('--output', type=str, default='datasets/benign_60m',
                       help='Output directory')
    parser.add_argument('--tranco', type=str, default='datasets/url_analysis/top-1m.csv',
                       help='Path to Tranco top-1m.csv')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry = HashRegistry()
    
    print("=" * 60)
    print("Generating 60M Benign Samples")
    print("=" * 60)
    
    # 1. URL Variations (25M - increased)
    print("\n[1/8] Loading Tranco domains...")
    domains = load_tranco_domains(Path(args.tranco))
    print(f"Loaded {len(domains)} domains")
    
    print("\n[2/8] Generating URL variations (25M)...")
    write_jsonl(output_dir / 'urls_25m.jsonl', 
                generate_urls(domains, 25_000_000), 25_000_000, registry, "URLs")
    
    # 2. SQL Queries (8M - increased)
    print("\n[3/8] Generating SQL queries (8M)...")
    write_jsonl(output_dir / 'sql_8m.jsonl',
                generate_sql(8_000_000), 8_000_000, registry, "SQL")
    
    # 3. Shell Commands (5M - increased)
    print("\n[4/8] Generating shell commands (5M)...")
    write_jsonl(output_dir / 'shell_5m.jsonl',
                generate_shell(5_000_000), 5_000_000, registry, "Shell")
    
    # 4. API Calls (8M - increased)
    print("\n[5/8] Generating API calls (8M)...")
    write_jsonl(output_dir / 'api_8m.jsonl',
                generate_api_calls(8_000_000), 8_000_000, registry, "API")
    
    # 5. Code Snippets (6M - increased)
    print("\n[6/8] Generating code snippets (6M)...")
    write_jsonl(output_dir / 'code_6m.jsonl',
                generate_code_snippets(6_000_000), 6_000_000, registry, "Code")
    
    # 6. Log Entries (4M - increased)
    print("\n[7/8] Generating log entries (4M)...")
    write_jsonl(output_dir / 'logs_4m.jsonl',
                generate_logs(4_000_000), 4_000_000, registry, "Logs")
    
    # 7. Configs + Text (4M)
    print("\n[8/8] Generating configs and text (4M)...")
    write_jsonl(output_dir / 'configs_2m.jsonl',
                generate_configs(2_000_000), 2_000_000, registry, "Configs")
    write_jsonl(output_dir / 'text_2m.jsonl',
                generate_text(2_000_000), 2_000_000, registry, "Text")
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    
    total = 0
    for f in output_dir.glob('*.jsonl'):
        size = f.stat().st_size / 1e6
        lines = sum(1 for _ in open(f))
        total += lines
        print(f"  {f.name}: {lines:,} samples ({size:.1f} MB)")
    
    print(f"\nTotal: {total:,} samples")
    print(f"Unique hashes tracked: {len(registry.seen):,}")


if __name__ == '__main__':
    main()
