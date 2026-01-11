#!/usr/bin/env python3
"""Retrain all models with specialized data for each model type."""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def run_training(script_path, model_name, description, timeout=None):
    """Run a training script."""
    print(f"\n{'='*80}")
    print(f" Training: {model_name}")
    print(f" {description}")
    print(f" Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False
    
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"  ✓ Completed in {timedelta(seconds=int(elapsed))}")
            return True
        else:
            print(f"  ✗ Failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    base = Path(__file__).parent.parent
    src = base / "src"
    training = base / "src" / "training"
    
    start_time = time.time()
    results = {}
    
    print("=" * 80)
    print(" SPECIALIZED MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nEach model will be trained on its specialized data:")
    print("  - Payload CNN: Security payloads + curated benign")
    print("  - URL CNN: Malicious URLs + benign URLs")
    print("  - Network Intrusion: Network attack data + MAWI benign")
    print("  - Fraud Detection: Fraud transactions + fraud benign")
    print("  - Timeseries LSTM: Attack traffic + normal traffic")
    print("  - Host Behavior: Malware patterns + host benign")
    
    # 1. Payload CNN - Security payloads + curated benign
    print("\n" + "="*80)
    print("[1/6] PAYLOAD CNN - Character-level injection detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 94.5M security payloads (SQL, XSS, command injection)")
    print("  Benign: 71M curated benign + 121M live benign")
    print("  Total: ~286M samples")
    
    payload_script = training / "train_payload.py"
    results['payload_cnn'] = run_training(
        payload_script,
        "Payload CNN",
        "Character-level CNN for injection attacks",
        timeout=28800  # 8 hours
    )
    
    # 2. URL CNN - URL analysis
    print("\n" + "="*80)
    print("[2/6] URL CNN - Malicious URL detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 5.2M malicious URLs (phishing, malware)")
    print("  Benign: 11M benign URLs (Tranco + Common Crawl)")
    print("  Total: 16.2M samples")
    
    url_script = training / "train_url_cnn.py"
    results['url_cnn'] = run_training(
        url_script,
        "URL CNN",
        "Character-level CNN for URL classification",
        timeout=10800  # 3 hours
    )
    
    # 3. Network Intrusion - Network traffic analysis
    print("\n" + "="*80)
    print("[3/6] NETWORK INTRUSION - Traffic anomaly detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 5.5M attack samples (DoS, Probe, R2L, U2R)")
    print("  Benign: 1M MAWI normal traffic")
    print("  Total: 6.5M samples")
    
    network_script = src / "train_network_intrusion.py"
    results['network_intrusion'] = run_training(
        network_script,
        "Network Intrusion RandomForest",
        "41-feature network traffic classifier",
        timeout=3600  # 1 hour
    )
    
    # 4. Fraud Detection - Transaction analysis
    print("\n" + "="*80)
    print("[4/6] FRAUD DETECTION - Transaction anomaly detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 785K fraudulent transactions")
    print("  Benign: 5M normal transactions")
    print("  Total: 5.8M samples")
    
    fraud_script = src / "train_fraud_detection.py"
    results['fraud_detection'] = run_training(
        fraud_script,
        "Fraud Detection XGBoost",
        "Transaction pattern classifier",
        timeout=1800  # 30 min
    )
    
    # 5. Timeseries LSTM - Temporal pattern analysis
    print("\n" + "="*80)
    print("[5/6] TIMESERIES LSTM - Temporal anomaly detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 250K attack sequences (DDoS, portscan)")
    print("  Benign: 1.25M normal traffic sequences")
    print("  Total: 1.5M samples")
    
    timeseries_script = training / "train_timeseries_lstm.py"
    results['timeseries_lstm'] = run_training(
        timeseries_script,
        "Timeseries LSTM",
        "Temporal sequence classifier",
        timeout=3600  # 1 hour
    )
    
    # 6. Host Behavior - Malware detection
    print("\n" + "="*80)
    print("[6/6] HOST BEHAVIOR - Malware pattern detection")
    print("="*80)
    print("Data sources:")
    print("  Malicious: 500K malware patterns")
    print("  Benign: 5M normal host behavior")
    print("  Total: 5.5M samples")
    
    host_script = src / "train_host_behavior.py"
    results['host_behavior'] = run_training(
        host_script,
        "Host Behavior RandomForest",
        "Process/memory pattern classifier",
        timeout=1800  # 30 min
    )
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print(" TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("\nModel Status:")
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model:25s}: {status}")
    
    print("\n" + "=" * 80)
    print(" DATA UTILIZATION")
    print("=" * 80)
    print("Total dataset: 322.6M samples (54.51 GB)")
    print("  - Benign: 215.6M (66.8%)")
    print("  - Malicious: 107.1M (33.2%)")
    print("  - Ratio: 2:1 (optimal for production)")
    print("=" * 80)
    
    if failed > 0:
        print("\n⚠ Some models failed to train. Check logs above.")
        return 1
    else:
        print("\n✓ All models trained successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
