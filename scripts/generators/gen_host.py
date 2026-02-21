#!/usr/bin/env python3
"""Generate 10M host behavior samples (37 features matching scaler)."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler, append_to_jsonl

def gen_benign_host():
    """Generate benign host behavior (37 features)."""
    return {
        "pslist_nproc": random.randint(50, 150),
        "pslist_nppid": random.randint(30, 100),
        "pslist_avg_threads": random.uniform(2, 10),
        "pslist_nprocs64bit": random.randint(40, 120),
        "pslist_avg_handlers": random.uniform(100, 500),
        "dlllist_ndlls": random.randint(500, 2000),
        "dlllist_avg_dlls_per_proc": random.uniform(20, 80),
        "handles_nhandles": random.randint(5000, 20000),
        "handles_avg_handles_per_proc": random.uniform(100, 400),
        "handles_nport": random.randint(10, 100),
        "handles_nfile": random.randint(100, 1000),
        "handles_nevent": random.randint(200, 2000),
        "handles_ndesktop": random.randint(1, 10),
        "handles_nkey": random.randint(500, 3000),
        "handles_nthread": random.randint(100, 500),
        "handles_ndirectory": random.randint(50, 300),
        "handles_nsemaphore": random.randint(10, 100),
        "handles_ntimer": random.randint(5, 50),
        "handles_nsection": random.randint(50, 300),
        "handles_nmutant": random.randint(20, 200),
        "ldrmodules_not_in_load": random.randint(0, 5),
        "ldrmodules_not_in_init": random.randint(0, 5),
        "ldrmodules_not_in_mem": random.randint(0, 3),
        "malfind_ninjections": 0,
        "malfind_commitcharge": random.randint(0, 1000),
        "malfind_protection": random.uniform(0, 0.1),
        "malfind_uniqueinjections": 0,
        "psxview_not_in_pslist": 0,
        "psxview_not_in_eprocess_pool": 0,
        "psxview_not_in_ethread_pool": 0,
        "psxview_not_in_pspcid_list": 0,
        "psxview_not_in_csrss_handles": random.randint(0, 2),
        "psxview_not_in_session": 0,
        "psxview_not_in_deskthrd": random.randint(0, 3),
        "modules_nmodules": random.randint(100, 300),
        "svcscan_nservices": random.randint(100, 300),
        "svcscan_kernel_drivers": random.randint(50, 150),
        "label": 0
    }

def gen_malicious_host():
    """Generate malicious host behavior (37 features)."""
    host = gen_benign_host()
    host["label"] = 1
    
    malware_type = random.choice(['spyware', 'ransomware', 'trojan', 'rootkit'])
    
    if malware_type == 'spyware':
        host["handles_nfile"] = random.randint(2000, 5000)
        host["handles_nkey"] = random.randint(5000, 10000)
        host["dlllist_ndlls"] = random.randint(2500, 4000)
    elif malware_type == 'ransomware':
        host["handles_nfile"] = random.randint(3000, 8000)
        host["malfind_ninjections"] = random.randint(1, 5)
        host["malfind_uniqueinjections"] = random.randint(1, 3)
    elif malware_type == 'trojan':
        host["pslist_nproc"] = random.randint(160, 250)
        host["ldrmodules_not_in_load"] = random.randint(5, 15)
        host["ldrmodules_not_in_init"] = random.randint(5, 15)
        host["malfind_ninjections"] = random.randint(2, 8)
    else:  # rootkit
        host["psxview_not_in_pslist"] = random.randint(1, 5)
        host["psxview_not_in_eprocess_pool"] = random.randint(1, 3)
        host["ldrmodules_not_in_mem"] = random.randint(3, 10)
        host["malfind_protection"] = random.uniform(0.5, 1.0)
    
    return host

def generate_samples(n_benign: int, n_malicious: int, output_dir: Path, batch_size: int = 100000):
    """Generate and save samples."""
    total = n_benign + n_malicious
    tracker = ProgressTracker(total, "Host")
    
    output_path = output_dir / "host_expansion.jsonl"
    
    batch = []
    for _ in range(n_benign):
        batch.append(gen_benign_host())
        if len(batch) >= batch_size:
            append_to_jsonl(output_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    
    for _ in range(n_malicious):
        batch.append(gen_malicious_host())
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
    parser = argparse.ArgumentParser(description="Generate host behavior samples")
    parser.add_argument("--total", type=int, default=10_000_000, help="Total samples")
    parser.add_argument("--output", type=Path, default=Path("datasets/host_behavior"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} benign + {n_malicious:,} malicious = {args.total:,} total")
    
    generate_samples(n_benign, n_malicious, args.output)

if __name__ == "__main__":
    main()
