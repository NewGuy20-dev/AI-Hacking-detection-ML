#!/usr/bin/env python3
"""Generate 500k synthetic host behavior samples for malware detection."""
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Feature names based on CIC-MalMem-2022 dataset
FEATURES = [
    'pslist.nproc', 'pslist.nppid', 'pslist.avg_threads', 'pslist.nprocs64bit',
    'pslist.avg_handlers', 'dlllist.ndlls', 'dlllist.avg_dlls_per_proc',
    'handles.nhandles', 'handles.avg_handles_per_proc', 'handles.nport',
    'handles.nfile', 'handles.nevent', 'handles.ndesktop', 'handles.nkey',
    'handles.nthread', 'handles.ndirectory', 'handles.nsemaphore',
    'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem',
    'malfind.ninjections', 'malfind.commitCharge', 'malfind.protection',
    'malfind.uniqueInjections', 'psxview.not_in_pslist', 'psxview.not_in_eprocess_pool',
    'psxview.not_in_ethread_pool', 'psxview.not_in_pspcid_list',
    'modules.nmodules', 'svcscan.nservices', 'svcscan.kernel_drivers',
    'svcscan.fs_drivers', 'svcscan.process_services', 'svcscan.shared_process_services',
    'callbacks.ncallbacks', 'callbacks.nanonymous', 'callbacks.ngeneric'
]

MALWARE_TYPES = ['Spyware', 'Ransomware', 'Trojan', 'Rootkit', 'Backdoor', 'Worm']

def generate_benign():
    """Generate benign host behavior."""
    return {
        'pslist.nproc': random.randint(50, 150),
        'pslist.nppid': random.randint(30, 80),
        'pslist.avg_threads': random.uniform(3, 10),
        'pslist.nprocs64bit': random.randint(40, 120),
        'pslist.avg_handlers': random.uniform(100, 500),
        'dlllist.ndlls': random.randint(500, 2000),
        'dlllist.avg_dlls_per_proc': random.uniform(20, 60),
        'handles.nhandles': random.randint(10000, 50000),
        'handles.avg_handles_per_proc': random.uniform(100, 400),
        'handles.nport': random.randint(50, 300),
        'handles.nfile': random.randint(500, 3000),
        'handles.nevent': random.randint(1000, 5000),
        'handles.ndesktop': random.randint(5, 20),
        'handles.nkey': random.randint(2000, 10000),
        'handles.nthread': random.randint(200, 800),
        'handles.ndirectory': random.randint(100, 500),
        'handles.nsemaphore': random.randint(100, 500),
        'ldrmodules.not_in_load': 0,
        'ldrmodules.not_in_init': random.randint(0, 5),
        'ldrmodules.not_in_mem': 0,
        'malfind.ninjections': 0,
        'malfind.commitCharge': random.randint(0, 1000),
        'malfind.protection': random.randint(0, 2),
        'malfind.uniqueInjections': 0,
        'psxview.not_in_pslist': 0,
        'psxview.not_in_eprocess_pool': 0,
        'psxview.not_in_ethread_pool': 0,
        'psxview.not_in_pspcid_list': 0,
        'modules.nmodules': random.randint(100, 300),
        'svcscan.nservices': random.randint(150, 400),
        'svcscan.kernel_drivers': random.randint(100, 200),
        'svcscan.fs_drivers': random.randint(5, 20),
        'svcscan.process_services': random.randint(50, 150),
        'svcscan.shared_process_services': random.randint(20, 80),
        'callbacks.ncallbacks': random.randint(50, 150),
        'callbacks.nanonymous': random.randint(0, 5),
        'callbacks.ngeneric': random.randint(10, 50),
        'label': 0,
        'category': 'Benign'
    }

def generate_spyware():
    """Generate spyware behavior patterns."""
    sample = generate_benign()
    # Spyware: hidden processes, many handles, network activity
    sample['psxview.not_in_pslist'] = random.randint(1, 5)
    sample['handles.nport'] = random.randint(300, 800)  # Many network connections
    sample['handles.nfile'] = random.randint(3000, 8000)  # File access
    sample['ldrmodules.not_in_load'] = random.randint(1, 10)
    sample['malfind.ninjections'] = random.randint(1, 5)
    sample['callbacks.nanonymous'] = random.randint(5, 20)
    sample['label'] = 1
    sample['category'] = 'Spyware'
    return sample

def generate_ransomware():
    """Generate ransomware behavior patterns."""
    sample = generate_benign()
    # Ransomware: high file activity, encryption indicators
    sample['handles.nfile'] = random.randint(5000, 20000)  # Mass file access
    sample['pslist.avg_threads'] = random.uniform(15, 50)  # High CPU
    sample['malfind.ninjections'] = random.randint(2, 10)
    sample['malfind.commitCharge'] = random.randint(5000, 50000)
    sample['handles.nkey'] = random.randint(10000, 30000)  # Registry access
    sample['ldrmodules.not_in_init'] = random.randint(5, 20)
    sample['label'] = 1
    sample['category'] = 'Ransomware'
    return sample

def generate_trojan():
    """Generate trojan behavior patterns."""
    sample = generate_benign()
    # Trojan: hidden processes, network backdoor
    sample['psxview.not_in_pslist'] = random.randint(1, 3)
    sample['psxview.not_in_eprocess_pool'] = random.randint(0, 2)
    sample['handles.nport'] = random.randint(200, 600)
    sample['malfind.ninjections'] = random.randint(3, 15)
    sample['malfind.uniqueInjections'] = random.randint(1, 5)
    sample['ldrmodules.not_in_load'] = random.randint(2, 8)
    sample['ldrmodules.not_in_mem'] = random.randint(1, 5)
    sample['callbacks.nanonymous'] = random.randint(3, 15)
    sample['label'] = 1
    sample['category'] = 'Trojan'
    return sample

def generate_rootkit():
    """Generate rootkit behavior patterns."""
    sample = generate_benign()
    # Rootkit: kernel-level hiding, driver manipulation
    sample['psxview.not_in_pslist'] = random.randint(2, 8)
    sample['psxview.not_in_eprocess_pool'] = random.randint(1, 5)
    sample['psxview.not_in_ethread_pool'] = random.randint(1, 5)
    sample['psxview.not_in_pspcid_list'] = random.randint(1, 5)
    sample['svcscan.kernel_drivers'] = random.randint(200, 350)  # Extra drivers
    sample['callbacks.ncallbacks'] = random.randint(150, 300)
    sample['callbacks.nanonymous'] = random.randint(10, 50)
    sample['modules.nmodules'] = random.randint(300, 500)
    sample['ldrmodules.not_in_load'] = random.randint(5, 20)
    sample['ldrmodules.not_in_mem'] = random.randint(3, 15)
    sample['malfind.ninjections'] = random.randint(5, 20)
    sample['label'] = 1
    sample['category'] = 'Rootkit'
    return sample

def generate_backdoor():
    """Generate backdoor behavior patterns."""
    sample = generate_benign()
    # Backdoor: persistent network access, hidden services
    sample['handles.nport'] = random.randint(400, 1000)
    sample['svcscan.nservices'] = random.randint(400, 600)  # Extra services
    sample['psxview.not_in_pslist'] = random.randint(1, 4)
    sample['malfind.ninjections'] = random.randint(2, 8)
    sample['ldrmodules.not_in_load'] = random.randint(1, 5)
    sample['callbacks.nanonymous'] = random.randint(5, 25)
    sample['handles.nevent'] = random.randint(5000, 15000)
    sample['label'] = 1
    sample['category'] = 'Backdoor'
    return sample

def generate_worm():
    """Generate worm behavior patterns."""
    sample = generate_benign()
    # Worm: self-replication, network scanning
    sample['handles.nport'] = random.randint(500, 2000)  # Network scanning
    sample['pslist.nproc'] = random.randint(150, 300)  # Many processes
    sample['handles.nfile'] = random.randint(3000, 10000)
    sample['malfind.ninjections'] = random.randint(5, 25)
    sample['malfind.uniqueInjections'] = random.randint(3, 15)
    sample['pslist.avg_threads'] = random.uniform(10, 30)
    sample['ldrmodules.not_in_init'] = random.randint(5, 15)
    sample['label'] = 1
    sample['category'] = 'Worm'
    return sample

def main():
    output_path = Path(__file__).parent.parent / "datasets" / "host_behavior" / "synthetic_500k.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 500_000
    generators = [
        (generate_benign, 250_000),      # 50% benign
        (generate_spyware, 50_000),      # 10% spyware
        (generate_ransomware, 50_000),   # 10% ransomware
        (generate_trojan, 50_000),       # 10% trojan
        (generate_rootkit, 35_000),      # 7% rootkit
        (generate_backdoor, 35_000),     # 7% backdoor
        (generate_worm, 30_000),         # 6% worm
    ]
    
    print(f"Generating {total:,} host behavior samples...")
    print(f"Output: {output_path}")
    
    samples = []
    for gen_func, count in generators:
        name = gen_func.__name__.replace('generate_', '')
        for _ in tqdm(range(count), desc=name.capitalize()):
            samples.append(gen_func())
    
    random.shuffle(samples)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            # Round floats
            rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in sample.items()}
            f.write(json.dumps(rounded) + '\n')
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Generated {total:,} samples ({size_mb:.1f} MB)")
    print(f"  Benign: 250k")
    print(f"  Spyware: 50k, Ransomware: 50k, Trojan: 50k")
    print(f"  Rootkit: 35k, Backdoor: 35k, Worm: 30k")

if __name__ == "__main__":
    main()
