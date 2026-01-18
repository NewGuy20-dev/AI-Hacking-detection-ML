#!/usr/bin/env python3
"""Generate 500k diverse benign samples for false positive testing.

Memory-efficient streaming generation with parallel processing.

Usage: python scripts/generate_500k_benign_test.py --output datasets/fp_test_500k.jsonl
"""
import argparse
import json
import random
import string
import hashlib
import time
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import sys

# ============================================================================
# DATA POOLS
# ============================================================================
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", 
    "Linda", "William", "Elizabeth", "Emma", "Oliver", "Sophia", "Liam", "Carlos", 
    "Maria", "Ahmed", "Fatima", "Wei", "Yuki", "Raj", "Priya", "Ivan", "Anna"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
    "Davis", "Rodriguez", "Martinez", "Lee", "Nguyen", "Kim", "Patel", "Chen"]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "company.com"]

TOPICS = ["science", "history", "technology", "art", "music", "sports", "medicine",
    "economics", "culture", "travel", "education", "environment", "psychology"]

SUBJECTS = ["quantum mechanics", "renewable energy", "artificial intelligence", 
    "climate change", "genetic engineering", "space exploration", "neural networks",
    "sustainable agriculture", "marine biology", "cognitive science", "nanotechnology"]

EMOTICONS = [":-)", ":-(", ";-)", ":D", "XD", "<3", ":P", "^_^", "O_O", "♥", "★", 
    "🎉", "👍", "😀", "😃", "🎊", "👌", "✓", "→", "←"]

SEARCH_QUERIES = ["best restaurants near me", "weather forecast", "how to learn python",
    "cheap flights", "recipe chocolate cake", "movie showtimes", "latest news",
    "online shopping deals", "fitness tips", "home renovation", "best laptop 2024"]

COMMENTS = ["Great product!", "Fast shipping!", "Good quality.", "Would buy again.",
    "Excellent service!", "Works perfectly.", "Very satisfied.", "Amazing!",
    "Love it!", "Quick delivery!", "As advertised.", "Perfect fit!"]


def fetch_wikipedia_extracts(count=50):
    """Fetch random Wikipedia extracts via API."""
    extracts = []
    try:
        url = "https://en.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=50&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            titles = [p["title"] for p in data.get("query", {}).get("random", [])]
        
        for title in titles[:count]:
            try:
                enc_title = urllib.parse.quote(title)
                url = f"https://en.wikipedia.org/w/api.php?action=query&titles={enc_title}&prop=extracts&exintro=1&explaintext=1&format=json"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    pages = data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = page.get("extract", "")
                        if extract and len(extract) > 50:
                            extracts.append((extract[:500], title))
            except:
                continue
    except Exception as e:
        print(f"Wikipedia API error: {e}", file=sys.stderr)
    return extracts


# ============================================================================
# GENERATORS (yield one sample at a time for memory efficiency)
# ============================================================================
def gen_name():
    mid = f" {random.choice(string.ascii_uppercase)}." if random.random() < 0.3 else ""
    suffix = random.choice(["", "", "", " Jr.", " Sr.", " III", " PhD", " MD"]) 
    # Add random ID to ensure uniqueness
    uid = f" ({random.randint(1,99999)})" if random.random() < 0.3 else ""
    return f"{random.choice(FIRST_NAMES)}{mid} {random.choice(LAST_NAMES)}{suffix}{uid}"

def gen_email():
    f, l = random.choice(FIRST_NAMES).lower(), random.choice(LAST_NAMES).lower()
    num = str(random.randint(1, 99999))  # Always add number for uniqueness
    sep = random.choice([".", "_", "", "-"])
    extra = random.choice(["", str(random.randint(0,99)), random.choice(string.ascii_lowercase)])
    return f"{f}{sep}{l}{num}{extra}@{random.choice(DOMAINS)}"

def gen_phone():
    area = random.randint(200, 999)
    pre = random.randint(200, 999)
    line = random.randint(1000, 9999)
    ext = f" ext.{random.randint(100,9999)}" if random.random() < 0.3 else ""
    formats = [
        f"({area}) {pre}-{line}{ext}",
        f"{area}-{pre}-{line}{ext}",
        f"+1 {area} {pre} {line}{ext}",
        f"{area}.{pre}.{line}{ext}",
        f"1-{area}-{pre}-{line}{ext}",
    ]
    return random.choice(formats)

def gen_wiki_sentence():
    topic = random.choice(TOPICS)
    subject = random.choice(SUBJECTS)
    year = random.randint(1800, 2024)
    pct = random.randint(1, 99)
    num = random.randint(2, 500)
    adj = random.choice(["significant", "important", "crucial", "notable", "remarkable", "substantial"])
    verb = random.choice(["studied", "researched", "analyzed", "examined", "investigated", "explored"])
    
    templates = [
        f"The {topic} of {subject} has been {verb} extensively since {year}.",
        f"{subject.title()} is {adj} in modern {topic} with {num} applications.",
        f"Research from {year} shows {subject} affects approximately {pct}% of cases.",
        f"The history of {subject} in {topic} dates back to {year}.",
        f"In {year}, experts discovered that {subject} plays a {adj} role in {topic}.",
        f"Studies indicate {subject} has {num} distinct characteristics in {topic}.",
        f"The relationship between {subject} and {topic} was first documented in {year}.",
        f"Approximately {pct}% of {topic} research focuses on {subject}.",
        f"Since {year}, {subject} has become increasingly {adj} to {topic}.",
        f"There are {num} known applications of {subject} in {topic} as of {year}.",
    ]
    return random.choice(templates)

def gen_json():
    n = random.randint(1, 9999999)  # Larger range
    n2 = random.randint(1, 99999)
    name = gen_name()
    r = random.random()
    
    if r < 0.1:
        obj = {"status": random.choice(["ok", "success", "pending", "active", "completed"]), "code": random.choice([200, 201, 204]), "id": n}
    elif r < 0.2:
        obj = {"id": n, "name": name, "active": random.choice([True, False]), "score": n2}
    elif r < 0.3:
        obj = {"items": [random.randint(1,1000) for _ in range(random.randint(1,5))], "total": n, "page": n2 % 100}
    elif r < 0.4:
        obj = {"user": {"id": n, "name": name, "email": gen_email(), "level": n2 % 100}}
    elif r < 0.5:
        obj = {"timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z", "event_id": n}
    elif r < 0.6:
        obj = {"page": n % 1000, "limit": random.choice([10,20,50,100]), "offset": n2, "query_id": n}
    elif r < 0.7:
        obj = {"error": None, "message": f"Request {n} processed", "duration_ms": n2}
    elif r < 0.8:
        obj = {"version": f"{random.randint(1,9)}.{random.randint(0,99)}.{random.randint(0,999)}", "build": n}
    elif r < 0.9:
        obj = {"lat": round(random.uniform(-90,90), 6), "lng": round(random.uniform(-180,180), 6), "accuracy": n2}
    else:
        obj = {"price": round(random.uniform(0.01, 9999.99), 2), "currency": random.choice(["USD", "EUR", "GBP", "JPY"]), "tx_id": n}
    return json.dumps(obj)

def gen_xml():
    n = random.randint(1, 9999999)
    n2 = random.randint(1, 99999)
    name = gen_name()
    r = random.random()
    
    if r < 0.2:
        return f"<user id=\"{n}\"><name>{name}</name><email>{gen_email()}</email><score>{n2}</score></user>"
    elif r < 0.4:
        return f"<item sku=\"SKU{n}\"><title>Product {n2}</title><price>{random.randint(1,9999)}.{random.randint(0,99):02d}</price><stock>{random.randint(0,1000)}</stock></item>"
    elif r < 0.6:
        return f"<config version=\"{random.randint(1,9)}.{random.randint(0,99)}\" id=\"{n}\"><setting name=\"opt_{n2}\">{random.choice(['true','false','auto'])}</setting></config>"
    elif r < 0.8:
        return f"<response status=\"ok\" id=\"{n}\"><data count=\"{n2}\">{name}</data></response>"
    else:
        return f"<order id=\"{n}\"><customer>{name}</customer><total>{random.randint(10,50000)}</total><items>{n2}</items></order>"

def gen_url():
    domains = ["example.com", "test.org", "mysite.net", "company.io", "app.dev", "service.co", "platform.com", "site.org"]
    n = random.randint(1, 9999999)
    n2 = random.randint(1, 99999)
    paths = [
        f"/user/{n}/profile",
        f"/api/v{random.randint(1,5)}/resource/{n}",
        f"/blog/post-{n}",
        f"/item/{n}?ref={n2}",
        f"/search?q=query_{n}&page={n2 % 100}",
        f"/category/{random.choice(TOPICS)}/{n}",
        f"/download/{n}/file_{n2}.zip",
        f"/order/{n}/status",
        f"/product/{n}?variant={n2}",
        f"/docs/v{random.randint(1,5)}/section-{n2}",
    ]
    proto = random.choice(["https://", "http://", "https://www."])
    return f"{proto}{random.choice(domains)}{random.choice(paths)}"

def gen_filepath():
    win_dirs = ["Users", "Documents", "Downloads", "Projects", "Desktop", "AppData", "Program Files"]
    unix_dirs = ["home", "var", "etc", "opt", "usr", "tmp", "data", "srv"]
    exts = [".txt", ".json", ".log", ".md", ".py", ".js", ".html", ".css", ".xml", ".csv"]
    n = random.randint(1, 99999)
    
    if random.random() < 0.5:
        return f"C:\\{random.choice(win_dirs)}\\{random.choice(FIRST_NAMES)}_{n}\\{random.choice(win_dirs)}\\file_{n}{random.choice(exts)}"
    else:
        return f"/{random.choice(unix_dirs)}/{random.choice(FIRST_NAMES).lower()}_{n}/{random.choice(unix_dirs)}/data_{n}{random.choice(exts)}"

def gen_code_python():
    funcs = ['process', 'calc', 'validate', 'transform', 'handle', 'parse', 'load', 'save', 'fetch', 'update']
    vars = ['data', 'result', 'items', 'values', 'config', 'params', 'output', 'response', 'records', 'entries']
    n = random.randint(1, 1000)
    f, v = random.choice(funcs), random.choice(vars)
    op = random.choice(['+', '-', '*', '//', '%'])
    val = random.randint(1, 100)
    
    templates = [
        f"def {f}_{n}({v}):\n    return {v} {op} {val}",
        f"for i in range({n}):\n    {v}[i] = i {op} {val}",
        f"if {v}_{n} is not None:\n    {v}_{n}.{random.choice(funcs)}()",
        f"{v}_{n} = [{f}(x) for x in range({n})]",
        f"class {f.title()}{n}:\n    def __init__(self, {v}):\n        self.{v} = {v}",
        f"try:\n    {v} = {f}({n})\nexcept Exception:\n    {v} = {val}",
        f"while {v} < {n}:\n    {v} {op}= {val}",
        f"import {random.choice(['os', 'sys', 'json', 'time', 'math'])}\n{v} = {n}",
        f"lambda x: x {op} {val} if x > {n} else x",
        f"@decorator\ndef {f}_{n}():\n    pass",
    ]
    return random.choice(templates)

def gen_code_js():
    funcs = ['process', 'handle', 'fetch', 'render', 'update', 'validate', 'transform', 'parse']
    vars = ['data', 'result', 'items', 'config', 'state', 'props', 'response', 'payload']
    n = random.randint(1, 1000)
    f, v = random.choice(funcs), random.choice(vars)
    op = random.choice(['+', '-', '*', '/'])
    
    templates = [
        f"function {f}_{n}({v}) {{ return {v} {op} {n}; }}",
        f"const {v}_{n} = {vars[1]}.map(x => x {op} {n});",
        f"if ({v}_{n} !== null) {{ {v}_{n}.{f}(); }}",
        f"const {{ {v} }} = require('./{f}_{n}');",
        f"async function {f}_{n}() {{ await fetch('/api/{v}'); }}",
        f"export const {v}_{n} = ({f}) => {f} {op} {n};",
        f"let {v} = {n};\n{v} {op}= {random.randint(1,100)};",
        f"class {f.title()}{n} {{ constructor() {{ this.{v} = {n}; }} }}",
        f"const {v}_{n} = [...{vars[0]}].filter(x => x > {n});",
        f"({v}) => {{ console.log({v} {op} {n}); }}",
    ]
    return random.choice(templates)

def gen_code_sql():
    tables = ['users', 'orders', 'products', 'customers', 'logs', 'sessions', 'events', 'items']
    cols = ['id', 'name', 'email', 'status', 'created_at', 'value', 'type', 'count']
    t, c = random.choice(tables), random.choice(cols)
    n = random.randint(1, 10000)
    op = random.choice(['=', '>', '<', '>=', '<=', '!='])
    
    templates = [
        f"SELECT {c}, {random.choice(cols)} FROM {t} WHERE {random.choice(cols)} {op} {n}",
        f"INSERT INTO {t} ({c}, {random.choice(cols)}) VALUES ('{gen_name()}', {n})",
        f"UPDATE {t} SET {c} = {n} WHERE id = {random.randint(1,1000)}",
        f"DELETE FROM {t} WHERE {c} {op} {n}",
        f"SELECT COUNT(*) FROM {t} WHERE {c} {op} {n} GROUP BY {random.choice(cols)}",
        f"SELECT * FROM {t} ORDER BY {c} DESC LIMIT {random.randint(10,100)}",
        f"SELECT {t}.{c}, {random.choice(tables)}.{random.choice(cols)} FROM {t} JOIN {random.choice(tables)} ON {t}.id = {random.choice(tables)}.{t}_id",
        f"SELECT DISTINCT {c} FROM {t} WHERE {random.choice(cols)} IS NOT NULL",
        f"SELECT AVG({c}), MAX({c}), MIN({c}) FROM {t}",
        f"CREATE INDEX idx_{t}_{c}_{n} ON {t}({c})",
    ]
    return random.choice(templates)


# ============================================================================
# CATEGORY GENERATORS (streaming)
# ============================================================================
def generate_wikipedia_batch(count, wiki_cache=None):
    """Generate Wikipedia-style content."""
    # Use cached Wikipedia extracts if available
    if wiki_cache:
        for text, title in wiki_cache:
            yield {"text": text, "category": "wikipedia", "source": title}
    
    # Fill rest with synthetic
    generated = len(wiki_cache) if wiki_cache else 0
    for _ in range(count - generated):
        r = random.random()
        if r < 0.5:
            text = gen_wiki_sentence()
        else:
            text = " ".join(gen_wiki_sentence() for _ in range(random.randint(2, 4)))
        yield {"text": text, "category": "wikipedia", "source": "synthetic"}


def generate_realworld_batch(count):
    """Generate real-world text samples."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next week"]
    times = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM", "4:00 PM", "11:00 AM", "1:00 PM"]
    carriers = ["FedEx", "UPS", "USPS", "DHL", "Amazon"]
    
    for _ in range(count):
        r = random.random()
        n = random.randint(10000, 9999999)
        n2 = random.randint(1, 99999)
        name = gen_name()
        
        if r < 0.15:
            text, src = name, "name"
        elif r < 0.28:
            text, src = gen_email(), "email"
        elif r < 0.40:
            text, src = gen_phone(), "phone"
        elif r < 0.70:
            templates = [
                f"Hi {name}, following up on conversation #{n}.",
                f"Dear {name}, thank you for inquiry #{n}.",
                f"Order #{n} shipped via {random.choice(carriers)} - tracking {n2}.",
                f"Reminder: Appointment #{n} on {random.choice(days)} at {random.choice(times)}.",
                f"Account #{n} updated successfully at {random.choice(times)}.",
                f"Document #{n} attached for review - ref {n2}.",
                f"Meeting with {name} scheduled for {random.choice(days)}, ref #{n}.",
                f"Reservation #{n} confirmed for {name}.",
                f"Ticket #{n} resolved - case {n2} closed.",
                f"Invoice #{n} - Amount due: ${random.randint(10,9999)}.{random.randint(0,99):02d}",
                f"Welcome {name}! Your member ID is {n}.",
                f"Password reset requested for account #{n}.",
                f"Subscription #{n} renewed until 2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}.",
            ]
            text, src = random.choice(templates), "email_body"
        elif r < 0.85:
            q = random.choice(SEARCH_QUERIES)
            mods = ["2024", "2025", "near me", "best", "cheap", "free", "online", "reviews", "top 10", str(n2)]
            text, src = f"{q} {random.choice(mods)}", "search"
        else:
            c = random.choice(COMMENTS)
            text, src = f"{c} Order #{n}", "comment"
        yield {"text": text, "category": "realworld", "source": src}


def generate_edgecases_batch(count):
    """Generate edge case samples."""
    unicode_samples = ["Café", "résumé", "naïve", "日本語", "한국어", "Привет", "مرحبا", 
        "中文", "Ñoño", "Größe", "北京", "東京", "Ελληνικά", "हिंदी", "ไทย"]
    special = ["$", "%", "&", "#", "@", "!", "?", "+", "=", "*", "^", "~"]
    
    for _ in range(count):
        r = random.random()
        n = random.randint(1, 9999)
        
        if r < 0.2:
            text = f"{random.choice(EMOTICONS)} {random.choice(FIRST_NAMES)} {random.choice(EMOTICONS)}"
            src = "emoticon"
        elif r < 0.4:
            words = random.sample(unicode_samples, random.randint(2, 4))
            text = f"{' '.join(words)} {n}"
            src = "unicode"
        elif r < 0.6:
            s1, s2 = random.choice(special), random.choice(special)
            text = f"{s1}{random.randint(1,999)}{s2}{random.randint(1,999)}{random.choice(special)}"
            src = "special"
        elif r < 0.8:
            parts = [random.choice(FIRST_NAMES), random.choice(EMOTICONS), 
                    str(n), random.choice(unicode_samples)]
            random.shuffle(parts)
            text = " ".join(parts)
            src = "mixed"
        else:
            # Random alphanumeric with punctuation
            chars = string.ascii_letters + string.digits + " .,!?-_"
            text = "".join(random.choice(chars) for _ in range(random.randint(20, 100)))
            src = "random"
        yield {"text": text, "category": "edgecases", "source": src}


def generate_code_batch(count):
    """Generate code snippet samples."""
    for _ in range(count):
        r = random.random()
        if r < 0.3:
            text, src = gen_code_python(), "python"
        elif r < 0.55:
            text, src = gen_code_js(), "javascript"
        elif r < 0.75:
            text, src = gen_code_sql(), "sql"
        else:
            # Bash with variety
            n = random.randint(1, 999)
            cmds = [
                f"#!/bin/bash\necho 'Process {n} started'",
                f"for f in *.txt; do cat \"$f\" >> output_{n}.log; done",
                f"if [ -f \"file_{n}.txt\" ]; then rm \"file_{n}.txt\"; fi",
                f"grep -r 'pattern_{n}' /path/to/dir_{n}",
                f"find . -name '*.log' -mtime +{n % 30} -delete",
                f"mkdir -p /tmp/dir_{n} && cd /tmp/dir_{n}",
                f"curl -s https://api.example.com/v{n % 5}/data",
                f"tar -czf backup_{n}.tar.gz /data/dir_{n}",
                f"chmod {random.choice(['644', '755', '600'])} script_{n}.sh",
                f"export VAR_{n}=\"value_{random.randint(1,100)}\"",
            ]
            text, src = random.choice(cmds), "bash"
        yield {"text": text, "category": "code", "source": src}


def generate_structured_batch(count):
    """Generate structured data samples."""
    for _ in range(count):
        r = random.random()
        if r < 0.25:
            text, src = gen_json(), "json"
        elif r < 0.45:
            text, src = gen_xml(), "xml"
        elif r < 0.6:
            text, src = f"id,name,email\n1,{gen_name()},{gen_email()}", "csv"
        elif r < 0.8:
            text, src = gen_url(), "url"
        else:
            text, src = gen_filepath(), "filepath"
        yield {"text": text, "category": "structured", "source": src}


CATEGORY_GENERATORS = {
    "wikipedia": generate_wikipedia_batch,
    "realworld": generate_realworld_batch,
    "edgecases": generate_edgecases_batch,
    "code": generate_code_batch,
    "structured": generate_structured_batch,
}


# ============================================================================
# STREAMING WRITER & MAIN
# ============================================================================
def process_category(args):
    """Process a single category (for parallel execution)."""
    cat_name, count, seen_file, wiki_cache = args
    
    # Load seen hashes for deduplication
    seen = set()
    if Path(seen_file).exists():
        with open(seen_file, 'r') as f:
            seen = set(line.strip() for line in f)
    
    results = []
    duplicates = 0
    
    if cat_name == "wikipedia":
        gen = generate_wikipedia_batch(count, wiki_cache)
    else:
        gen = CATEGORY_GENERATORS[cat_name](count)
    
    for sample in gen:
        h = hashlib.md5(sample["text"].encode()).hexdigest()[:16]
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        results.append(sample)
        if len(results) >= count:
            break
    
    # Save seen hashes
    with open(seen_file, 'w') as f:
        f.write('\n'.join(seen))
    
    return cat_name, results, duplicates


def main():
    parser = argparse.ArgumentParser(description="Generate 500k benign samples")
    parser.add_argument("--output", "-o", default="datasets/fp_test_500k.jsonl")
    parser.add_argument("--count", "-n", type=int, default=500000)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--batch-size", "-b", type=int, default=10000)
    parser.add_argument("--skip-wiki-api", action="store_true", help="Skip Wikipedia API calls")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".tmp_benign_gen"
    temp_dir.mkdir(exist_ok=True)
    
    per_category = args.count // 5
    
    print(f"{'='*60}")
    print(f" Benign Sample Generator - {args.count:,} samples")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Per category: {per_category:,}")
    print(f"Workers: {args.workers}")
    print()
    
    start_time = time.time()
    
    # Fetch Wikipedia content (optional)
    wiki_cache = []
    if not args.skip_wiki_api:
        print("Fetching Wikipedia extracts...")
        wiki_cache = fetch_wikipedia_extracts(100)
        print(f"  Fetched {len(wiki_cache)} Wikipedia extracts")
    
    # Process categories
    total_written = 0
    total_duplicates = 0
    stats = {}
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for cat_name in CATEGORY_GENERATORS:
            cat_start = time.time()
            seen_file = temp_dir / f"{cat_name}_seen.txt"
            
            print(f"\nGenerating {cat_name}...")
            
            # Generate in batches for memory efficiency
            cat_count = 0
            cat_dups = 0
            seen = set()
            
            if cat_name == "wikipedia":
                gen = generate_wikipedia_batch(per_category, wiki_cache)
            else:
                gen = CATEGORY_GENERATORS[cat_name](per_category)
            
            batch = []
            for sample in gen:
                h = hashlib.md5(sample["text"].encode()).hexdigest()[:16]
                if h in seen:
                    cat_dups += 1
                    continue
                seen.add(h)
                
                record = {
                    "id": total_written + len(batch),
                    "text": sample["text"],
                    "category": sample["category"],
                    "source": sample["source"],
                    "length": len(sample["text"]),
                }
                batch.append(record)
                
                # Write batch
                if len(batch) >= args.batch_size:
                    for rec in batch:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += len(batch)
                    cat_count += len(batch)
                    batch = []
                    
                    # Progress
                    elapsed = time.time() - start_time
                    rate = total_written / elapsed if elapsed > 0 else 0
                    eta = (args.count - total_written) / rate if rate > 0 else 0
                    print(f"  Progress: {total_written:,}/{args.count:,} ({rate:.0f}/s, ETA: {eta:.0f}s)", end='\r')
                
                if cat_count >= per_category:
                    break
            
            # Write remaining
            for rec in batch:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_written += len(batch)
            cat_count += len(batch)
            
            cat_elapsed = time.time() - cat_start
            stats[cat_name] = cat_count
            total_duplicates += cat_dups
            print(f"  ✓ {cat_name}: {cat_count:,} samples in {cat_elapsed:.1f}s (dups: {cat_dups})")
    
    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f" COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {total_written:,}")
    print(f"Duplicates filtered: {total_duplicates:,}")
    print(f"Time: {elapsed:.1f}s ({total_written/elapsed:.0f} samples/s)")
    print(f"Output: {output_path} ({file_size:.1f} MB)")
    print(f"\nCategory breakdown:")
    for cat, count in stats.items():
        pct = count / total_written * 100 if total_written > 0 else 0
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    
    # Validation metrics
    print(f"\nValidation:")
    print(f"  Unique samples: {total_written:,}")
    print(f"  Avg length: ~{sum(stats.values()) // len(stats) if stats else 0}")
    print(f"  Categories: {len(stats)}")


if __name__ == "__main__":
    main()
