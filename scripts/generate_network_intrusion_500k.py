#!/usr/bin/env python3
"""Generate 500k synthetic network intrusion samples."""
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# NSL-KDD feature names (41 features)
FEATURES = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'smtp', 'ftp', 'ssh', 'dns', 'telnet', 'pop3', 'imap', 'https', 'other']
FLAGS = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH']

def generate_normal():
    """Generate normal traffic pattern."""
    return {
        'duration': random.randint(0, 1000),
        'protocol_type': random.choice(PROTOCOLS),
        'service': random.choice(SERVICES),
        'flag': 'SF',  # Normal completion
        'src_bytes': random.randint(100, 10000),
        'dst_bytes': random.randint(100, 50000),
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': random.randint(0, 3),
        'num_failed_logins': 0,
        'logged_in': random.choice([0, 1]),
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': random.randint(0, 5),
        'num_shells': 0,
        'num_access_files': random.randint(0, 3),
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': random.randint(1, 50),
        'srv_count': random.randint(1, 50),
        'serror_rate': random.uniform(0, 0.1),
        'srv_serror_rate': random.uniform(0, 0.1),
        'rerror_rate': random.uniform(0, 0.1),
        'srv_rerror_rate': random.uniform(0, 0.1),
        'same_srv_rate': random.uniform(0.8, 1.0),
        'diff_srv_rate': random.uniform(0, 0.2),
        'srv_diff_host_rate': random.uniform(0, 0.3),
        'dst_host_count': random.randint(1, 255),
        'dst_host_srv_count': random.randint(1, 255),
        'dst_host_same_srv_rate': random.uniform(0.5, 1.0),
        'dst_host_diff_srv_rate': random.uniform(0, 0.3),
        'dst_host_same_src_port_rate': random.uniform(0, 0.5),
        'dst_host_srv_diff_host_rate': random.uniform(0, 0.3),
        'dst_host_serror_rate': random.uniform(0, 0.1),
        'dst_host_srv_serror_rate': random.uniform(0, 0.1),
        'dst_host_rerror_rate': random.uniform(0, 0.1),
        'dst_host_srv_rerror_rate': random.uniform(0, 0.1),
        'label': 0,
        'attack_type': 'normal'
    }

def generate_dos():
    """Generate DoS attack pattern."""
    return {
        'duration': 0,
        'protocol_type': random.choice(['tcp', 'icmp']),
        'service': random.choice(['http', 'ecr_i', 'private', 'other']),
        'flag': random.choice(['S0', 'REJ', 'SF']),
        'src_bytes': random.randint(0, 1000),
        'dst_bytes': 0,
        'land': random.choice([0, 1]),
        'wrong_fragment': random.randint(0, 3),
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': random.randint(100, 511),  # High count
        'srv_count': random.randint(1, 50),
        'serror_rate': random.uniform(0.8, 1.0),  # High error rate
        'srv_serror_rate': random.uniform(0.8, 1.0),
        'rerror_rate': random.uniform(0, 0.3),
        'srv_rerror_rate': random.uniform(0, 0.3),
        'same_srv_rate': random.uniform(0.9, 1.0),
        'diff_srv_rate': random.uniform(0, 0.1),
        'srv_diff_host_rate': random.uniform(0, 0.1),
        'dst_host_count': random.randint(200, 255),
        'dst_host_srv_count': random.randint(1, 50),
        'dst_host_same_srv_rate': random.uniform(0, 0.3),
        'dst_host_diff_srv_rate': random.uniform(0, 0.1),
        'dst_host_same_src_port_rate': random.uniform(0.9, 1.0),
        'dst_host_srv_diff_host_rate': random.uniform(0, 0.1),
        'dst_host_serror_rate': random.uniform(0.8, 1.0),
        'dst_host_srv_serror_rate': random.uniform(0.8, 1.0),
        'dst_host_rerror_rate': random.uniform(0, 0.2),
        'dst_host_srv_rerror_rate': random.uniform(0, 0.2),
        'label': 1,
        'attack_type': 'dos'
    }

def generate_probe():
    """Generate Probe/Scan attack pattern."""
    return {
        'duration': random.randint(0, 10),
        'protocol_type': random.choice(['tcp', 'udp', 'icmp']),
        'service': random.choice(['private', 'other', 'eco_i', 'ecr_i']),
        'flag': random.choice(['SF', 'S0', 'REJ', 'RSTR']),
        'src_bytes': random.randint(0, 500),
        'dst_bytes': random.randint(0, 500),
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': random.randint(1, 100),
        'srv_count': random.randint(1, 20),
        'serror_rate': random.uniform(0, 0.5),
        'srv_serror_rate': random.uniform(0, 0.5),
        'rerror_rate': random.uniform(0.5, 1.0),  # High reject rate
        'srv_rerror_rate': random.uniform(0.5, 1.0),
        'same_srv_rate': random.uniform(0, 0.3),  # Low - scanning many services
        'diff_srv_rate': random.uniform(0.7, 1.0),  # High - different services
        'srv_diff_host_rate': random.uniform(0.5, 1.0),
        'dst_host_count': random.randint(1, 100),
        'dst_host_srv_count': random.randint(1, 50),
        'dst_host_same_srv_rate': random.uniform(0, 0.3),
        'dst_host_diff_srv_rate': random.uniform(0.5, 1.0),
        'dst_host_same_src_port_rate': random.uniform(0, 0.3),
        'dst_host_srv_diff_host_rate': random.uniform(0.5, 1.0),
        'dst_host_serror_rate': random.uniform(0, 0.5),
        'dst_host_srv_serror_rate': random.uniform(0, 0.5),
        'dst_host_rerror_rate': random.uniform(0.5, 1.0),
        'dst_host_srv_rerror_rate': random.uniform(0.5, 1.0),
        'label': 1,
        'attack_type': 'probe'
    }

def generate_r2l():
    """Generate R2L (Remote to Local) attack pattern."""
    return {
        'duration': random.randint(0, 5000),
        'protocol_type': 'tcp',
        'service': random.choice(['ftp', 'telnet', 'smtp', 'pop3', 'imap']),
        'flag': random.choice(['SF', 'S0', 'REJ']),
        'src_bytes': random.randint(100, 5000),
        'dst_bytes': random.randint(0, 2000),
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': random.randint(0, 10),
        'num_failed_logins': random.randint(1, 5),  # Failed logins
        'logged_in': random.choice([0, 1]),
        'num_compromised': random.randint(0, 5),
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': random.randint(0, 3),
        'num_shells': 0,
        'num_access_files': random.randint(0, 5),
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': random.choice([0, 1]),
        'count': random.randint(1, 30),
        'srv_count': random.randint(1, 30),
        'serror_rate': random.uniform(0, 0.3),
        'srv_serror_rate': random.uniform(0, 0.3),
        'rerror_rate': random.uniform(0, 0.3),
        'srv_rerror_rate': random.uniform(0, 0.3),
        'same_srv_rate': random.uniform(0.8, 1.0),
        'diff_srv_rate': random.uniform(0, 0.2),
        'srv_diff_host_rate': random.uniform(0, 0.3),
        'dst_host_count': random.randint(1, 100),
        'dst_host_srv_count': random.randint(1, 100),
        'dst_host_same_srv_rate': random.uniform(0.5, 1.0),
        'dst_host_diff_srv_rate': random.uniform(0, 0.3),
        'dst_host_same_src_port_rate': random.uniform(0, 0.5),
        'dst_host_srv_diff_host_rate': random.uniform(0, 0.3),
        'dst_host_serror_rate': random.uniform(0, 0.3),
        'dst_host_srv_serror_rate': random.uniform(0, 0.3),
        'dst_host_rerror_rate': random.uniform(0, 0.3),
        'dst_host_srv_rerror_rate': random.uniform(0, 0.3),
        'label': 1,
        'attack_type': 'r2l'
    }

def generate_u2r():
    """Generate U2R (User to Root) attack pattern."""
    return {
        'duration': random.randint(0, 500),
        'protocol_type': 'tcp',
        'service': random.choice(['telnet', 'ftp', 'ssh', 'other']),
        'flag': 'SF',
        'src_bytes': random.randint(100, 3000),
        'dst_bytes': random.randint(100, 3000),
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': random.randint(5, 30),  # High hot indicators
        'num_failed_logins': random.randint(0, 3),
        'logged_in': 1,
        'num_compromised': random.randint(1, 10),  # Compromised
        'root_shell': random.choice([0, 1]),  # May get root
        'su_attempted': random.choice([0, 1, 2]),  # su attempts
        'num_root': random.randint(0, 10),
        'num_file_creations': random.randint(1, 10),
        'num_shells': random.randint(0, 3),
        'num_access_files': random.randint(1, 10),
        'num_outbound_cmds': random.randint(0, 5),
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': random.randint(1, 20),
        'srv_count': random.randint(1, 20),
        'serror_rate': random.uniform(0, 0.1),
        'srv_serror_rate': random.uniform(0, 0.1),
        'rerror_rate': random.uniform(0, 0.1),
        'srv_rerror_rate': random.uniform(0, 0.1),
        'same_srv_rate': random.uniform(0.9, 1.0),
        'diff_srv_rate': random.uniform(0, 0.1),
        'srv_diff_host_rate': random.uniform(0, 0.1),
        'dst_host_count': random.randint(1, 50),
        'dst_host_srv_count': random.randint(1, 50),
        'dst_host_same_srv_rate': random.uniform(0.8, 1.0),
        'dst_host_diff_srv_rate': random.uniform(0, 0.2),
        'dst_host_same_src_port_rate': random.uniform(0, 0.3),
        'dst_host_srv_diff_host_rate': random.uniform(0, 0.2),
        'dst_host_serror_rate': random.uniform(0, 0.1),
        'dst_host_srv_serror_rate': random.uniform(0, 0.1),
        'dst_host_rerror_rate': random.uniform(0, 0.1),
        'dst_host_srv_rerror_rate': random.uniform(0, 0.1),
        'label': 1,
        'attack_type': 'u2r'
    }

def main():
    output_path = Path(__file__).parent.parent / "datasets" / "network_intrusion" / "synthetic_500k.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 500_000
    # 50% normal, 50% attacks (balanced among attack types)
    generators = [
        (generate_normal, 250_000),
        (generate_dos, 100_000),
        (generate_probe, 75_000),
        (generate_r2l, 50_000),
        (generate_u2r, 25_000),
    ]
    
    print(f"Generating {total:,} network intrusion samples...")
    print(f"Output: {output_path}")
    
    samples = []
    for gen_func, count in generators:
        name = gen_func.__name__.replace('generate_', '')
        for _ in tqdm(range(count), desc=name):
            samples.append(gen_func())
    
    random.shuffle(samples)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Generated {total:,} samples ({size_mb:.1f} MB)")
    print(f"  Normal: 250k, DoS: 100k, Probe: 75k, R2L: 50k, U2R: 25k")

if __name__ == "__main__":
    main()
