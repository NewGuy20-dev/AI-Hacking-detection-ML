#!/usr/bin/env python3
"""Test payload model FP rate on 5M fresh benign samples (not in training data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random
import string

# Generators for fresh data
def gen_sentences(n):
    """Generate natural sentences."""
    subjects = ["The user", "A customer", "Our team", "The system", "This product", 
                "The service", "Your order", "The report", "Our company", "The manager"]
    verbs = ["is processing", "has completed", "will review", "needs to update",
             "successfully handled", "is waiting for", "has been approved", "requires"]
    objects = ["the request", "your account", "the payment", "this transaction",
               "the document", "your submission", "the application", "the data"]
    for i in range(n):
        yield f"{subjects[i%len(subjects)]} {verbs[i%len(verbs)]} {objects[i%len(objects)]} #{i:07d}"

def gen_emails(n):
    """Generate email addresses."""
    names = ["john", "jane", "mike", "sarah", "alex", "emma", "david", "lisa", "chris", "anna"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "example.org"]
    for i in range(n):
        yield f"{names[i%len(names)]}.{random.choice(string.ascii_lowercase)}{i:06d}@{domains[i%len(domains)]}"

def gen_paths(n):
    """Generate file paths."""
    dirs = ["/home/user", "/var/log", "/opt/app", "C:\\Users\\Admin", "./data", "../config"]
    exts = [".txt", ".log", ".json", ".csv", ".py", ".js", ".xml", ".yaml"]
    for i in range(n):
        yield f"{dirs[i%len(dirs)]}/file_{i:08d}{exts[i%len(exts)]}"

def gen_products(n):
    """Generate product descriptions."""
    adjs = ["Premium", "Professional", "Basic", "Advanced", "Standard", "Deluxe", "Ultra"]
    items = ["Widget", "Service", "Package", "Solution", "Tool", "System", "Kit"]
    for i in range(n):
        yield f"{adjs[i%len(adjs)]} {items[i%len(items)]} v{i%100}.{i%10} - ${(i%1000)+9.99:.2f}"

def gen_logs(n):
    """Generate log entries."""
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    services = ["api", "web", "db", "cache", "auth", "worker"]
    msgs = ["Request processed", "Connection established", "Cache hit", "Task completed", "User authenticated"]
    for i in range(n):
        ts = f"2024-{(i%12)+1:02d}-{(i%28)+1:02d} {i%24:02d}:{i%60:02d}:{i%60:02d}"
        yield f"[{ts}] [{levels[i%len(levels)]}] [{services[i%len(services)]}] {msgs[i%len(msgs)]} id={i:08d}"

def gen_json(n):
    """Generate JSON snippets."""
    for i in range(n):
        yield f'{{"id": {i}, "name": "item_{i:06d}", "active": {str(i%2==0).lower()}, "count": {i%1000}}}'

def gen_urls(n):
    """Generate benign URLs."""
    domains = ["example.com", "test.org", "mysite.net", "company.io", "service.co"]
    paths = ["/home", "/about", "/products", "/api/v1", "/user", "/search", "/docs"]
    for i in range(n):
        yield f"https://www.{domains[i%len(domains)]}{paths[i%len(paths)]}?id={i:08d}"

def gen_code(n):
    """Generate code snippets."""
    templates = [
        "def func_{0}(x): return x * {1}",
        "const val_{0} = {1};",
        "for i in range({1}): print(i + {0})",
        "if (x_{0} > {1}) {{ return true; }}",
        "let arr_{0} = [{1}, {2}, {3}];",
        "class Item{0}: pass",
        "import module_{0}",
        "# Comment for task {0}",
    ]
    for i in range(n):
        t = templates[i % len(templates)]
        yield t.format(i, i%100, i%50, i%25)

def gen_names(n):
    """Generate names with apostrophes and special chars."""
    firsts = ["John", "Mary", "O'Brien", "McDonald", "Jean-Pierre", "María", "José", "François"]
    lasts = ["Smith", "O'Connor", "MacDonald", "St. Claire", "Van der Berg", "De la Cruz"]
    for i in range(n):
        yield f"{firsts[i%len(firsts)]} {lasts[i%len(lasts)]} #{i:06d}"

def gen_math(n):
    """Generate math expressions."""
    for i in range(n):
        a, b = i % 1000, (i * 7) % 1000
        ops = [f"{a} + {b} = {a+b}", f"{a} * {b} = {a*b}", f"{a} / {max(b,1)} = {a/max(b,1):.2f}",
               f"sqrt({a}) = {a**0.5:.2f}", f"{a}^2 = {a**2}", f"{a} % {max(b,1)} = {a%max(b,1)}"]
        yield ops[i % len(ops)]

def gen_addresses(n):
    """Generate addresses."""
    streets = ["Main St", "Oak Ave", "Park Blvd", "First St", "Elm Dr", "Cedar Ln"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle"]
    for i in range(n):
        yield f"{(i%9999)+1} {streets[i%len(streets)]}, {cities[i%len(cities)]} {10000+(i%90000)}"


def load_model(model_path):
    """Load payload CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try TorchScript first, then regular checkpoint
    try:
        model = torch.jit.load(model_path, map_location=device)
    except:
        from torch_models.payload_cnn import PayloadCNN
        model = PayloadCNN(vocab_size=256, embed_dim=128, num_filters=256, max_len=500)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        model.to(device)
    
    model.eval()
    return model, device


def encode_batch(texts, max_len=500):
    """Encode texts to tensor."""
    batch = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        encoded = [ord(c) % 256 for c in text[:max_len]]
        batch[i, :len(encoded)] = encoded
    return torch.tensor(batch)


def test_fp():
    base = Path(__file__).parent.parent
    model_path = base / "models" / "payload_cnn.pt"
    
    print("=" * 60)
    print(" FALSE POSITIVE TEST - 5M FRESH BENIGN SAMPLES")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_path}")
    
    # Load model
    print("\n--- Loading Model ---")
    model, device = load_model(model_path)
    print(f"Device: {device}")
    
    # Generators (500k each = 5M total)
    generators = [
        ("Sentences", gen_sentences, 500_000),
        ("Emails", gen_emails, 500_000),
        ("File Paths", gen_paths, 500_000),
        ("Products", gen_products, 500_000),
        ("Log Entries", gen_logs, 500_000),
        ("JSON", gen_json, 500_000),
        ("URLs", gen_urls, 500_000),
        ("Code Snippets", gen_code, 500_000),
        ("Names", gen_names, 500_000),
        ("Math/Addresses", gen_math, 250_000),
    ]
    # Add addresses to reach 5M
    generators.append(("Addresses", gen_addresses, 250_000))
    
    total_samples = sum(g[2] for g in generators)
    print(f"\n--- Testing {total_samples:,} samples ---")
    
    batch_size = 2048
    total_fp = 0
    total_tested = 0
    fp_examples = []
    category_stats = {}
    
    with torch.no_grad():
        for cat_name, gen_func, count in generators:
            cat_fp = 0
            cat_tested = 0
            batch = []
            
            pbar = tqdm(gen_func(count), total=count, desc=cat_name, leave=False)
            for text in pbar:
                batch.append(text)
                
                if len(batch) >= batch_size:
                    inputs = encode_batch(batch).to(device)
                    outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                    
                    fps = outputs > 0.5
                    cat_fp += fps.sum()
                    cat_tested += len(batch)
                    
                    # Save FP examples
                    if fps.any() and len(fp_examples) < 100:
                        for j, (is_fp, score) in enumerate(zip(fps, outputs)):
                            if is_fp and len(fp_examples) < 100:
                                fp_examples.append((cat_name, batch[j][:80], score))
                    
                    batch = []
                    pbar.set_postfix(fp_rate=f"{cat_fp/max(cat_tested,1)*100:.3f}%")
            
            # Process remaining
            if batch:
                inputs = encode_batch(batch).to(device)
                outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                fps = outputs > 0.5
                cat_fp += fps.sum()
                cat_tested += len(batch)
            
            fp_rate = cat_fp / max(cat_tested, 1) * 100
            category_stats[cat_name] = (cat_fp, cat_tested, fp_rate)
            total_fp += cat_fp
            total_tested += cat_tested
            
            print(f"  {cat_name}: {cat_fp:,}/{cat_tested:,} FP ({fp_rate:.3f}%)")
    
    # Summary
    overall_fp_rate = total_fp / total_tested * 100
    
    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print(f"Total Samples: {total_tested:,}")
    print(f"False Positives: {total_fp:,}")
    print(f"FP Rate: {overall_fp_rate:.4f}%")
    print(f"Target: <2-3%")
    print(f"Status: {'✓ PASS' if overall_fp_rate < 3 else '✗ FAIL'}")
    
    # Category breakdown
    print("\n--- Category Breakdown ---")
    for cat, (fp, tested, rate) in sorted(category_stats.items(), key=lambda x: -x[1][2]):
        status = "✓" if rate < 3 else "⚠" if rate < 5 else "✗"
        print(f"  {status} {cat}: {rate:.3f}% ({fp:,}/{tested:,})")
    
    # FP examples
    if fp_examples:
        print(f"\n--- Sample False Positives ({len(fp_examples)}) ---")
        for cat, text, score in fp_examples[:20]:
            print(f"  [{cat}] ({score:.3f}) {text}")
    
    # Save results
    results_file = base / "evaluation" / f"fp_test_5m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        f.write(f"FP Test Results - {datetime.now().isoformat()}\n")
        f.write(f"Total: {total_tested:,} samples\n")
        f.write(f"FP: {total_fp:,} ({overall_fp_rate:.4f}%)\n\n")
        for cat, (fp, tested, rate) in category_stats.items():
            f.write(f"{cat}: {fp}/{tested} ({rate:.3f}%)\n")
        f.write("\nFP Examples:\n")
        for cat, text, score in fp_examples:
            f.write(f"[{cat}] ({score:.3f}) {text}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return overall_fp_rate < 3


if __name__ == "__main__":
    success = test_fp()
    sys.exit(0 if success else 1)
