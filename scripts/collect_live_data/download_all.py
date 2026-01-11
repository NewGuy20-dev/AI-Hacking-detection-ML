#!/usr/bin/env python3
"""
Master script to download/generate all live benign data sources.
Includes feature extractors to match each model's expected format.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent.parent / "datasets" / "live_benign"

# Data sources organized by model type
DATA_SOURCES = {
    'payload_cnn': [
        ("Wikipedia Text (20M paragraphs)", "download_wikipedia.py", "Text for payload model"),
        ("Stack Overflow (50M posts)", "download_stackoverflow.py", "Code/text for payload model"),
        ("GitHub Archive (100M snippets)", "download_github_archive.py", "Code snippets - NO REPO CLONING"),
        ("Reddit Comments (100M)", "download_reddit.py", "User text for payload model"),
        ("Enron Emails (500k)", "download_enron.py", "Email text for payload model"),
    ],
    'url_cnn': [
        ("Common Crawl URLs (50-100M)", "download_common_crawl.py", "URLs for URL model"),
    ],
    'network_intrusion': [
        ("MAWI Network Traces (10M flows)", "download_mawi.py", "KDD-41 format network flows"),
    ],
    'fraud_detection': [
        ("Benign Transactions (5M)", "generate_fraud_benign.py", "PCA-format transaction features"),
    ],
    'host_behavior': [
        ("Benign Host Behavior (5M)", "generate_host_behavior.py", "38-feature memory analysis format"),
    ],
    'timeseries_lstm': [
        ("Benign Timeseries (1M sequences)", "generate_timeseries_benign.py", "(60, 4) sequences for LSTM"),
    ],
}

def print_menu():
    print("=" * 70)
    print("LIVE BENIGN DATA COLLECTOR")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nData sources by model:\n")
    
    idx = 1
    flat_list = []
    for model, sources in DATA_SOURCES.items():
        print(f"  [{model.upper()}]")
        for name, script, desc in sources:
            print(f"    {idx}. {name}")
            print(f"       -> {desc}")
            flat_list.append((name, script, model))
            idx += 1
        print()
    
    return flat_list

def run_script(script_name: str):
    script_path = SCRIPTS_DIR / script_name
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path)])
    else:
        print(f"Script not found: {script_path}")

def show_summary():
    print("\n" + "=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    
    if not OUTPUT_DIR.exists():
        print("No output directory yet.")
        return
    
    total_samples = 0
    total_size = 0
    
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            if f.suffix == '.jsonl':
                lines = sum(1 for _ in open(f))
                total_samples += lines
                print(f"  {f.name}: {lines:,} samples ({size_mb:.1f} MB)")
            elif f.suffix == '.npy':
                import numpy as np
                data = np.load(f)
                total_samples += data.shape[0]
                print(f"  {f.name}: {data.shape[0]:,} sequences ({size_mb:.1f} MB)")
            else:
                print(f"  {f.name}: {size_mb:.1f} MB")
    
    print(f"\n  TOTAL: {total_samples:,} samples, {total_size:.1f} MB")

def main():
    flat_list = print_menu()
    
    print("Options:")
    print("  all      - Run ALL downloaders/generators")
    print("  payload  - Run all Payload CNN sources")
    print("  url      - Run URL CNN source")
    print("  network  - Run Network Intrusion source")
    print("  fraud    - Run Fraud Detection generator")
    print("  host     - Run Host Behavior generator")
    print("  time     - Run Timeseries generator")
    print("  1-11     - Run specific source")
    print("  summary  - Show output summary")
    print("  q        - Quit")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'summary':
        show_summary()
    elif choice == 'all':
        for name, script, model in flat_list:
            print(f"\n{'=' * 70}")
            print(f"Running: {name} ({model})")
            print("=" * 70)
            run_script(script)
        show_summary()
    elif choice == 'payload':
        for name, script, desc in DATA_SOURCES['payload_cnn']:
            print(f"\n{'=' * 70}")
            print(f"Running: {name}")
            print("=" * 70)
            run_script(script)
    elif choice == 'url':
        for name, script, desc in DATA_SOURCES['url_cnn']:
            run_script(script)
    elif choice == 'network':
        for name, script, desc in DATA_SOURCES['network_intrusion']:
            run_script(script)
    elif choice == 'fraud':
        for name, script, desc in DATA_SOURCES['fraud_detection']:
            run_script(script)
    elif choice == 'host':
        for name, script, desc in DATA_SOURCES['host_behavior']:
            run_script(script)
    elif choice == 'time':
        for name, script, desc in DATA_SOURCES['timeseries_lstm']:
            run_script(script)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(flat_list):
                name, script, model = flat_list[idx]
                print(f"\nRunning: {name} ({model})")
                run_script(script)
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid choice")

if __name__ == "__main__":
    main()
