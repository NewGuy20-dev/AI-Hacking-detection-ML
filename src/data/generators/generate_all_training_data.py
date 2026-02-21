#!/usr/bin/env python3
"""Master script to generate training data for all models."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("URL CNN - 5M malicious URLs", "generate_malicious_urls_5m.py"),
    ("Network Intrusion - 500k samples", "generate_network_intrusion_500k.py"),
    ("Fraud Detection - 500k samples", "generate_fraud_500k.py"),
    ("Timeseries - 500k sequences", "generate_timeseries_500k.py"),
    ("Host Behavior - 500k samples", "generate_host_behavior_500k.py"),
]

def main():
    base = Path(__file__).parent
    
    print("=" * 60)
    print(" GENERATING TRAINING DATA FOR ALL MODELS")
    print("=" * 60)
    
    for desc, script in SCRIPTS:
        print(f"\n--- {desc} ---")
        script_path = base / script
        if script_path.exists():
            subprocess.run([sys.executable, str(script_path)])
        else:
            print(f"  Script not found: {script}")
    
    print("\n" + "=" * 60)
    print(" COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
