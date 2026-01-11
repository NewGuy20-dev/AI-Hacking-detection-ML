#!/usr/bin/env python3
"""Generate realistic benign host behavior samples (38-feature format)."""

import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "host_behavior_benign.jsonl"
TARGET_SAMPLES = 5_000_000

PROFILES = {
    'idle': {'nproc': (40, 80), 'nppid': (20, 50), 'ndlls': (500, 1200), 'nhandles': (15000, 40000), 'injections': (0, 0)},
    'active': {'nproc': (80, 150), 'nppid': (40, 80), 'ndlls': (800, 2000), 'nhandles': (30000, 60000), 'injections': (0, 1)},
    'developer': {'nproc': (100, 200), 'nppid': (50, 100), 'ndlls': (1000, 2500), 'nhandles': (40000, 80000), 'injections': (0, 2)},
    'server': {'nproc': (50, 100), 'nppid': (25, 60), 'ndlls': (600, 1500), 'nhandles': (20000, 50000), 'injections': (0, 0)},
}

def generate_sample(profile_name: str) -> dict:
    p = PROFILES[profile_name]
    nproc = random.randint(*p['nproc'])
    nhandles = random.randint(*p['nhandles'])
    ndlls = random.randint(*p['ndlls'])
    
    return {
        'pslist.nproc': nproc, 'pslist.nppid': random.randint(*p['nppid']),
        'pslist.avg_threads': round(random.uniform(3, 12), 4), 'pslist.nprocs64bit': random.randint(30, nproc),
        'pslist.avg_handlers': round(random.uniform(100, 500), 4), 'dlllist.ndlls': ndlls,
        'dlllist.avg_dlls_per_proc': round(ndlls / max(nproc, 1), 4), 'handles.nhandles': nhandles,
        'handles.avg_handles_per_proc': round(nhandles / max(nproc, 1), 4),
        'handles.nport': random.randint(100, 400), 'handles.nfile': random.randint(500, 2000),
        'handles.nevent': random.randint(2000, 6000), 'handles.ndesktop': random.randint(10, 20),
        'handles.nkey': random.randint(1500, 7000), 'handles.nthread': random.randint(300, 900),
        'handles.ndirectory': random.randint(150, 500), 'handles.nsemaphore': random.randint(100, 600),
        'ldrmodules.not_in_load': random.randint(0, 2), 'ldrmodules.not_in_init': random.randint(0, 3),
        'ldrmodules.not_in_mem': random.randint(0, 2), 'malfind.ninjections': random.randint(*p['injections']),
        'malfind.commitCharge': random.randint(100, 800), 'malfind.protection': random.randint(0, 1),
        'malfind.uniqueInjections': random.randint(*p['injections']),
        'psxview.not_in_pslist': 0, 'psxview.not_in_eprocess_pool': 0,
        'psxview.not_in_ethread_pool': 0, 'psxview.not_in_pspcid_list': 0,
        'modules.nmodules': random.randint(100, 300), 'svcscan.nservices': random.randint(200, 450),
        'svcscan.kernel_drivers': random.randint(100, 200), 'svcscan.fs_drivers': random.randint(5, 20),
        'svcscan.process_services': random.randint(40, 130), 'svcscan.shared_process_services': random.randint(20, 70),
        'callbacks.ncallbacks': random.randint(50, 150), 'callbacks.nanonymous': random.randint(0, 10),
        'callbacks.ngeneric': random.randint(10, 40), 'label': 0, 'category': 'Benign'
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_SAMPLES:,} samples")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_SAMPLES:
            print(f"Already complete with {existing:,} samples!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    profiles = list(PROFILES.keys())
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in tqdm(range(TARGET_SAMPLES), desc="Generating"):
            f.write(json.dumps(generate_sample(random.choice(profiles))) + '\n')
    
    print(f"\nGenerated {TARGET_SAMPLES:,} samples -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
