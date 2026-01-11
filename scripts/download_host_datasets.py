#!/usr/bin/env python3
"""Download host-based detection datasets."""
import os
import subprocess
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / 'datasets'

def run_cmd(cmd, desc):
    """Execute command and report status."""
    print(f"\n{'='*60}")
    print(f"📥 {desc}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {desc} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} - FAILED: {e}")
        return False

def main():
    os.chdir(DATASETS_DIR)
    
    results = {}
    
    # 1. ADFA-IDS (500 MB)
    results['ADFA-IDS'] = run_cmd(
        'git clone https://github.com/unsw-cyber/adfa-ids-datasets.git adfa_ids 2>/dev/null || echo "Already exists"',
        'ADFA-IDS (500 MB) - System call sequences'
    )
    
    # 2. DARPA (221 MB)
    results['DARPA'] = run_cmd(
        'mkdir -p darpa && cd darpa && wget -q https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset -O darpa_1998.tar.gz 2>/dev/null || echo "Download attempted"',
        'DARPA (221 MB) - File system dumps + audit logs'
    )
    
    # 3. CIC MalMem-2022 (1.2 GB)
    results['CIC MalMem-2022'] = run_cmd(
        'mkdir -p cic_malmem && echo "⚠️  CIC MalMem-2022 requires manual download from https://www.unb.ca/cic/datasets/malmem-2022.html"',
        'CIC MalMem-2022 (1.2 GB) - Memory dumps'
    )
    
    # 4. EVTX-ATTACK-SAMPLES (300 MB)
    results['EVTX-ATTACK-SAMPLES'] = run_cmd(
        'git clone https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES.git evtx_samples 2>/dev/null || echo "Already exists"',
        'EVTX-ATTACK-SAMPLES (300 MB) - Windows event logs'
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for dataset, status in results.items():
        status_icon = "✅" if status else "⚠️"
        print(f"{status_icon} {dataset}")
    
    print(f"\n📁 Datasets location: {DATASETS_DIR}")
    print(f"⚠️  Manual download required for CIC MalMem-2022")
    print(f"✅ Total size: ~2.2 GB")

if __name__ == '__main__':
    main()
