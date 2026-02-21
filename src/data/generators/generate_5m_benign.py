#!/usr/bin/env python3
"""Generate 5 million diverse benign samples for comprehensive training."""
import argparse
import json
import random
import string
import hashlib
import time
from pathlib import Path

# Data pools
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "Emma", "Oliver", "Sophia", "Liam", "Carlos", "Maria",
    "Ahmed", "Fatima", "Wei", "Yuki", "Raj", "Priya", "Ivan", "Anna", "Chen", "Kim"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Lee", "Nguyen", "Kim", "Patel", "Chen", "Singh", "Wilson"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "company.com", "work.org",
    "mail.co", "proton.me", "icloud.com", "aol.com"]
TLDS = [".com", ".org", ".net", ".io", ".co", ".dev", ".app", ".xyz"]
WORDS = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "great", "best"]
PRODUCTS = ["laptop", "phone", "camera", "headphones", "watch", "tablet", "speaker", "monitor",
    "keyboard", "mouse", "charger", "cable", "case", "stand", "bag", "light"]
ADJECTIVES = ["great", "good", "best", "amazing", "excellent", "perfect", "wonderful", "fantastic",
    "awesome", "nice", "beautiful", "lovely", "quick", "fast", "easy", "simple"]
VERBS = ["love", "like", "recommend", "bought", "received", "ordered", "tried", "used", "works"]

def gen_sentence():
    length = random.randint(5, 15)
    return " ".join(random.choice(WORDS) for _ in range(length)).capitalize() + "."

def gen_email():
    f = random.choice(FIRST_NAMES).lower()
    l = random.choice(LAST_NAMES).lower()
    n = random.randint(1, 9999)
    sep = random.choice([".", "_", "", "-"])
    return f"{f}{sep}{l}{n}@{random.choice(DOMAINS)}"

def gen_url():
    domain = "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(5, 12)))
    path = "/".join(random.choice(WORDS) for _ in range(random.randint(0, 4)))
    return f"https://{domain}{random.choice(TLDS)}/{path}"

def gen_review():
    adj = random.choice(ADJECTIVES)
    verb = random.choice(VERBS)
    product = random.choice(PRODUCTS)
    templates = [
        f"{adj.capitalize()} {product}!",
        f"I {verb} this {product}",
        f"{adj.capitalize()}! Would recommend",
        f"Five stars! {adj.capitalize()} quality",
        f"Fast shipping, {adj} product",
        f"Exactly as described. {adj.capitalize()}!",
    ]
    return random.choice(templates)

def gen_code():
    var = random.choice(["x", "y", "data", "result", "value", "item", "count", "total"])
    n = random.randint(1, 1000)
    templates = [
        f"def func_{n}({var}): return {var}",
        f"for i in range({n}): print(i)",
        f"if {var} > {n}: return True",
        f"const {var} = {n};",
        f"let {var} = getData({n});",
        f"function test_{n}() {{ }}",
        f"class Item{n} {{ }}",
        f"import module_{n}",
    ]
    return random.choice(templates)

def gen_json():
    n = random.randint(1, 999999)
    templates = [
        f'{{"id": {n}, "status": "ok"}}',
        f'{{"name": "{random.choice(FIRST_NAMES)}", "value": {n}}}',
        f'{{"count": {n}, "page": {random.randint(1, 100)}}}',
        f'{{"success": true, "data": {n}}}',
    ]
    return random.choice(templates)

def gen_path():
    n = random.randint(1, 9999)
    if random.random() < 0.5:
        return f"C:\\Users\\{random.choice(FIRST_NAMES)}\\Documents\\file_{n}.txt"
    return f"/home/{random.choice(FIRST_NAMES).lower()}/data/file_{n}.log"

def gen_log():
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    n = random.randint(1000, 9999)
    return f"[{random.choice(levels)}] Process {n}: {random.choice(WORDS)} completed"

def gen_mixed():
    """Generate mixed content."""
    parts = [
        gen_sentence,
        gen_email,
        gen_review,
        lambda: f"Order #{random.randint(10000, 99999)}",
        lambda: f"ID: {random.randint(1000, 9999)}",
        lambda: f"${random.randint(1, 999)}.{random.randint(0, 99):02d}",
    ]
    return random.choice(parts)()

GENERATORS = [
    (gen_sentence, 0.25),
    (gen_email, 0.10),
    (gen_url, 0.10),
    (gen_review, 0.15),
    (gen_code, 0.10),
    (gen_json, 0.08),
    (gen_path, 0.07),
    (gen_log, 0.05),
    (gen_mixed, 0.10),
]

def generate_sample():
    r = random.random()
    cumulative = 0
    for gen, prob in GENERATORS:
        cumulative += prob
        if r < cumulative:
            return gen()
    return gen_sentence()

def main():
    parser = argparse.ArgumentParser(description="Generate 5M benign samples")
    parser.add_argument("--output", "-o", default="datasets/benign_5m.jsonl")
    parser.add_argument("--count", "-n", type=int, default=5_000_000)
    parser.add_argument("--batch-size", "-b", type=int, default=100_000)
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.count:,} benign samples...")
    print(f"Output: {output_path}")
    
    start = time.time()
    seen = set()
    written = 0
    duplicates = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        while written < args.count:
            batch = []
            for _ in range(args.batch_size):
                if written + len(batch) >= args.count:
                    break
                text = generate_sample()
                h = hashlib.md5(text.encode()).hexdigest()[:12]
                if h in seen:
                    duplicates += 1
                    continue
                seen.add(h)
                batch.append({"id": written + len(batch), "text": text})
            
            for item in batch:
                f.write(json.dumps(item) + "\n")
            written += len(batch)
            
            elapsed = time.time() - start
            rate = written / elapsed if elapsed > 0 else 0
            eta = (args.count - written) / rate if rate > 0 else 0
            print(f"  Progress: {written:,}/{args.count:,} ({rate:,.0f}/s, ETA: {eta:.0f}s)", end="\r")
    
    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / 1024 / 1024
    
    print(f"\n\nComplete!")
    print(f"  Samples: {written:,}")
    print(f"  Duplicates skipped: {duplicates:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  File size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
