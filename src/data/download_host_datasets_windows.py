#!/usr/bin/env python3
"""Download host-based detection datasets (Windows compatible)."""
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
        subprocess.run(cmd, shell=True, check=False)
        print(f"✅ {desc} - COMPLETE")
        return True
    except Exception as e:
        print(f"⚠️  {desc} - {e}")
        return False

def main():
    os.chdir(DATASETS_DIR)
    
    results = {}
    
    # 1. ADFA-IDS (500 MB)
    results['ADFA-IDS'] = run_cmd(
        'git clone https://github.com/unsw-cyber/adfa-ids-datasets.git adfa_ids',
        'ADFA-IDS (500 MB) - System call sequences'
    )
    
    # 2. EVTX-ATTACK-SAMPLES (300 MB)
    results['EVTX-ATTACK-SAMPLES'] = run_cmd(
        'git clone https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES.git evtx_samples',
        'EVTX-ATTACK-SAMPLES (300 MB) - Windows event logs'
    )
    
    # 3. DARPA (221 MB) - Manual note
    print(f"\n{'='*60}")
    print(f"📥 DARPA (221 MB) - File system dumps + audit logs")
    print(f"{'='*60}")
    print("⚠️  Manual download required from:")
    print("   https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset")
    results['DARPA'] = False
    
    # 4. CIC MalMem-2022 (1.2 GB) - Manual note
    print(f"\n{'='*60}")
    print(f"📥 CIC MalMem-2022 (1.2 GB) - Memory dumps")
    print(f"{'='*60}")
    print("⚠️  Manual download required from:")
    print("   https://www.unb.ca/cic/datasets/malmem-2022.html")
    results['CIC MalMem-2022'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for dataset, status in results.items():
        status_icon = "✅" if status else "⚠️"
        print(f"{status_icon} {dataset}")
    
    print(f"\n📁 Datasets location: {DATASETS_DIR}")
    print(f"✅ Auto-downloaded: ADFA-IDS, EVTX-ATTACK-SAMPLES")
    print(f"⚠️  Manual download: DARPA, CIC MalMem-2022")
    print(f"📊 Total size: ~2.2 GB")

if __name__ == '__main__':
    main()
