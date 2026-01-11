#!/usr/bin/env python3
"""Generate realistic benign network flows in KDD format locally."""

import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "mawi_network_kdd.jsonl"
TARGET_FLOWS = 5_000_000

SERVICES = ['http', 'https', 'ftp', 'ssh', 'smtp', 'dns', 'pop3', 'imap', 'telnet', 'other']
PROTOCOLS = ['tcp', 'udp', 'icmp']
FLAGS = ['SF', 'S0', 'S1', 'REJ', 'RSTO', 'SH', 'RSTR', 'S2', 'S3', 'OTH']

# Benign traffic profiles
PROFILES = {
    'web_browsing': {'service': 'http', 'duration': (0, 60), 'src_bytes': (100, 50000), 'dst_bytes': (1000, 500000)},
    'https': {'service': 'https', 'duration': (0, 120), 'src_bytes': (100, 30000), 'dst_bytes': (500, 300000)},
    'email': {'service': 'smtp', 'duration': (0, 30), 'src_bytes': (100, 10000), 'dst_bytes': (100, 5000)},
    'dns': {'service': 'dns', 'duration': (0, 1), 'src_bytes': (30, 100), 'dst_bytes': (50, 500)},
    'ssh': {'service': 'ssh', 'duration': (0, 3600), 'src_bytes': (100, 100000), 'dst_bytes': (100, 100000)},
    'ftp': {'service': 'ftp', 'duration': (0, 300), 'src_bytes': (100, 1000000), 'dst_bytes': (100, 1000000)},
}

def generate_benign_flow():
    """Generate a single benign network flow in KDD format."""
    profile_name = random.choice(list(PROFILES.keys()))
    p = PROFILES[profile_name]
    
    duration = random.randint(*p['duration'])
    src_bytes = random.randint(*p['src_bytes'])
    dst_bytes = random.randint(*p['dst_bytes'])
    
    # Benign characteristics
    count = random.randint(1, 50)
    srv_count = random.randint(1, count)
    
    return {
        'duration': duration,
        'protocol_type': 'tcp' if p['service'] in ['http', 'https', 'ssh', 'ftp', 'smtp'] else random.choice(PROTOCOLS),
        'service': p['service'],
        'flag': 'SF' if random.random() < 0.85 else random.choice(FLAGS),  # 85% normal completion
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': random.randint(0, 3),
        'num_failed_logins': 0,
        'logged_in': 1 if p['service'] in ['ssh', 'ftp', 'telnet'] and random.random() < 0.8 else 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': random.randint(0, 2) if p['service'] == 'ftp' else 0,
        'num_shells': 0,
        'num_access_files': random.randint(0, 3),
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': count,
        'srv_count': srv_count,
        'serror_rate': round(random.uniform(0, 0.1), 4),
        'srv_serror_rate': round(random.uniform(0, 0.1), 4),
        'rerror_rate': round(random.uniform(0, 0.05), 4),
        'srv_rerror_rate': round(random.uniform(0, 0.05), 4),
        'same_srv_rate': round(random.uniform(0.8, 1.0), 4),
        'diff_srv_rate': round(random.uniform(0, 0.2), 4),
        'srv_diff_host_rate': round(random.uniform(0, 0.1), 4),
        'dst_host_count': random.randint(1, 255),
        'dst_host_srv_count': random.randint(1, 255),
        'dst_host_same_srv_rate': round(random.uniform(0.8, 1.0), 4),
        'dst_host_diff_srv_rate': round(random.uniform(0, 0.1), 4),
        'dst_host_same_src_port_rate': round(random.uniform(0, 0.5), 4),
        'dst_host_srv_diff_host_rate': round(random.uniform(0, 0.1), 4),
        'dst_host_serror_rate': round(random.uniform(0, 0.05), 4),
        'dst_host_srv_serror_rate': round(random.uniform(0, 0.05), 4),
        'dst_host_rerror_rate': round(random.uniform(0, 0.05), 4),
        'dst_host_srv_rerror_rate': round(random.uniform(0, 0.05), 4),
        'label': 0,
        'attack_type': 'normal'
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_FLOWS:,} KDD-format flows")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE))
        if existing >= TARGET_FLOWS:
            print(f"Already complete with {existing:,} flows!")
            return
        print(f"Found {existing:,} partial. Restarting...")
    
    print("Generating benign network flows locally...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(TARGET_FLOWS), desc="Generating"):
            flow = generate_benign_flow()
            f.write(json.dumps(flow) + '\n')
    
    print(f"\n✓ Generated {TARGET_FLOWS:,} flows -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
