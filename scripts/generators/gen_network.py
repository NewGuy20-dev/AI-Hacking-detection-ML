#!/usr/bin/env python3
"""Generate 10M network flow samples (35 features matching scaler)."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler, append_to_jsonl

# Feature ranges based on KDD99/NSL-KDD
PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'pop3', 'imap', 'https', 'other']
FLAGS = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH']

def gen_benign_flow():
    """Generate benign network flow (35 features)."""
    return {
        "duration": random.uniform(0, 1000),
        "protocol_type": random.choice(PROTOCOLS),
        "service": random.choice(SERVICES),
        "flag": random.choice(['SF', 'S0']),  # Normal flags
        "src_bytes": random.randint(0, 50000),
        "dst_bytes": random.randint(0, 50000),
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": random.randint(0, 5),
        "num_failed_logins": 0,
        "logged_in": random.randint(0, 1),
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": random.randint(0, 3),
        "num_shells": 0,
        "num_access_files": random.randint(0, 2),
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": random.randint(1, 100),
        "srv_count": random.randint(1, 50),
        "serror_rate": random.uniform(0, 0.1),
        "srv_serror_rate": random.uniform(0, 0.1),
        "rerror_rate": random.uniform(0, 0.1),
        "srv_rerror_rate": random.uniform(0, 0.1),
        "same_srv_rate": random.uniform(0.8, 1.0),
        "diff_srv_rate": random.uniform(0, 0.2),
        "srv_diff_host_rate": random.uniform(0, 0.2),
        "dst_host_count": random.randint(1, 255),
        "dst_host_srv_count": random.randint(1, 255),
        "dst_host_same_srv_rate": random.uniform(0.8, 1.0),
        "dst_host_diff_srv_rate": random.uniform(0, 0.2),
        "label": 0
    }

def gen_malicious_flow():
    """Generate malicious network flow (35 features)."""
    attack_type = random.choice(['dos', 'probe', 'r2l', 'u2r'])
    flow = gen_benign_flow()
    flow["label"] = 1
    
    if attack_type == 'dos':
        flow["count"] = random.randint(200, 500)
        flow["srv_count"] = random.randint(100, 300)
        flow["serror_rate"] = random.uniform(0.5, 1.0)
        flow["same_srv_rate"] = random.uniform(0.9, 1.0)
        flow["flag"] = random.choice(['S0', 'REJ', 'RSTR'])
    elif attack_type == 'probe':
        flow["count"] = random.randint(50, 200)
        flow["diff_srv_rate"] = random.uniform(0.5, 1.0)
        flow["dst_host_diff_srv_rate"] = random.uniform(0.5, 1.0)
        flow["flag"] = random.choice(['REJ', 'RSTO', 'S0'])
    elif attack_type == 'r2l':
        flow["num_failed_logins"] = random.randint(1, 5)
        flow["hot"] = random.randint(5, 20)
        flow["num_compromised"] = random.randint(1, 10)
        flow["logged_in"] = 1
    else:  # u2r
        flow["root_shell"] = 1
        flow["su_attempted"] = random.randint(1, 3)
        flow["num_root"] = random.randint(1, 10)
        flow["num_shells"] = random.randint(1, 3)
    
    return flow

def generate_samples(n_benign: int, n_malicious: int, output_dir: Path, batch_size: int = 100000):
    """Generate and save samples."""
    total = n_benign + n_malicious
    tracker = ProgressTracker(total, "Network")
    
    output_path = output_dir / "network_expansion.jsonl"
    
    batch = []
    for _ in range(n_benign):
        batch.append(gen_benign_flow())
        if len(batch) >= batch_size:
            append_to_jsonl(output_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    
    for _ in range(n_malicious):
        batch.append(gen_malicious_flow())
        if len(batch) >= batch_size:
            append_to_jsonl(output_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    
    if batch:
        append_to_jsonl(output_path, iter(batch))
        tracker.update(len(batch))
    
    tracker.close()
    print(f"Saved: {output_path} ({total:,} samples)")

def main():
    parser = argparse.ArgumentParser(description="Generate network flow samples")
    parser.add_argument("--total", type=int, default=10_000_000, help="Total samples")
    parser.add_argument("--output", type=Path, default=Path("datasets/network_intrusion"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} benign + {n_malicious:,} malicious = {args.total:,} total")
    
    generate_samples(n_benign, n_malicious, args.output)

if __name__ == "__main__":
    main()
