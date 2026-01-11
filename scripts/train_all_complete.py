#!/usr/bin/env python3
"""
Complete retraining script for all 6 models with specialized data mapping.
Optimized for RTX 3050 + i5 12th Gen.
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Discord notifications
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1452715933398466782/Ajftu5_fHelFqifTRcZN3S7fCDddXPs89p9w8dTHX8pF1xUO59ckac_DyCTQsRKC1H8O"


class DiscordNotifier:
    """Send training notifications to Discord."""
    
    COLORS = {'info': 0x3498db, 'success': 0x2ecc71, 'warning': 0xf39c12, 'error': 0xe74c3c}
    
    def __init__(self, webhook_url: str = DISCORD_WEBHOOK):
        self.webhook_url = webhook_url
        self.enabled = HAS_REQUESTS and webhook_url
    
    def _send(self, embed: dict) -> bool:
        if not self.enabled:
            return False
        try:
            requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            return True
        except:
            return False
    
    def pipeline_started(self):
        """🚀 Pipeline started."""
        embed = {
            "title": "🚀 Complete Model Retraining Started",
            "color": self.COLORS['info'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Total Models", "value": "6", "inline": True},
                {"name": "Total Samples", "value": "322.6M", "inline": True},
                {"name": "Data Ratio", "value": "67% Benign : 33% Malicious", "inline": True},
            ]
        }
        return self._send(embed)
    
    def model_started(self, model_name: str, samples: str, mal: str, ben: str):
        """📦 Model training started."""
        embed = {
            "title": f"📦 Training: {model_name}",
            "color": self.COLORS['info'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Total Samples", "value": samples, "inline": True},
                {"name": "Malicious", "value": mal, "inline": True},
                {"name": "Benign", "value": ben, "inline": True},
            ]
        }
        return self._send(embed)
    
    def model_completed(self, model_name: str, success: bool, elapsed: str):
        """✅ Model completed."""
        embed = {
            "title": f"{'✅' if success else '❌'} {model_name}",
            "color": self.COLORS['success'] if success else self.COLORS['error'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Status", "value": "Success" if success else "Failed", "inline": True},
                {"name": "Time", "value": elapsed, "inline": True},
            ]
        }
        return self._send(embed)
    
    def pipeline_completed(self, total_time: str, successful: int, failed: int):
        """🎉 Pipeline completed."""
        embed = {
            "title": "🎉 All Training Complete!",
            "color": self.COLORS['success'] if failed == 0 else self.COLORS['warning'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Total Time", "value": total_time, "inline": True},
                {"name": "Successful", "value": str(successful), "inline": True},
                {"name": "Failed", "value": str(failed), "inline": True},
            ]
        }
        return self._send(embed)


def print_header():
    """Print pipeline header with dataset stats."""
    print("=" * 80)
    print(" COMPLETE MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n📊 DATASET OVERVIEW")
    print("  Total: 322.6M samples (54.51 GB)")
    print("  Benign: 215.6M (66.8%)")
    print("  Malicious: 107.1M (33.2%)")
    print("  Ratio: 2:1 (optimal for production)")
    print("=" * 80)


def print_model_info(num: int, name: str, total: str, mal: str, ben: str, desc: str):
    """Print model training info."""
    print(f"\n[{num}/6] {name}")
    print("-" * 80)
    print(f"  Description: {desc}")
    print(f"  Malicious: {mal}")
    print(f"  Benign: {ben}")
    print(f"  Total: {total}")
    print("-" * 80)


def run_training_script(script_path: Path, model_name: str) -> tuple[bool, str]:
    """Run a training script and return (success, elapsed_time)."""
    import time
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=False
        )
        elapsed = str(timedelta(seconds=int(time.time() - start)))
        return (result.returncode == 0, elapsed)
    except Exception as e:
        elapsed = str(timedelta(seconds=int(time.time() - start)))
        print(f"  ✗ Error: {e}")
        return (False, elapsed)


def main():
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all 6 models with specialized data')
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train: all, pytorch, sklearn, or comma-separated list')
    parser.add_argument('--no-discord', action='store_true', help='Disable Discord notifications')
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent
    training_dir = base / "src" / "training"
    src_dir = base / "src"
    
    # Initialize notifier
    notifier = DiscordNotifier() if not args.no_discord else None
    
    # Print header
    print_header()
    
    if notifier:
        notifier.pipeline_started()
    
    # Define all models with their data mapping
    models = {
        'payload': {
            'num': 1,
            'name': 'PAYLOAD CNN',
            'script': training_dir / 'train_payload.py',
            'total': '~286M samples',
            'malicious': '94.5M security payloads (SQL, XSS, command injection)',
            'benign': '192M curated + live benign',
            'desc': 'Character-level CNN for injection attack detection',
            'category': 'pytorch'
        },
        'url': {
            'num': 2,
            'name': 'URL CNN',
            'script': training_dir / 'train_url.py',
            'total': '16.2M samples',
            'malicious': '5.2M malicious URLs (phishing, malware)',
            'benign': '11M benign URLs (Tranco + Common Crawl)',
            'desc': 'Character-level CNN for URL classification',
            'category': 'pytorch'
        },
        'timeseries': {
            'num': 3,
            'name': 'TIMESERIES LSTM',
            'script': training_dir / 'train_timeseries.py',
            'total': '1.5M samples',
            'malicious': '250K attack sequences (DDoS, portscan)',
            'benign': '1.25M normal traffic sequences',
            'desc': 'LSTM for temporal anomaly detection',
            'category': 'pytorch'
        },
        'network': {
            'num': 4,
            'name': 'NETWORK INTRUSION RF',
            'script': src_dir / 'train_network_intrusion.py',
            'total': '6.5M samples',
            'malicious': '5.5M attack samples (DoS, Probe, R2L, U2R)',
            'benign': '1M MAWI normal traffic',
            'desc': 'RandomForest for 41-feature network traffic classification',
            'category': 'sklearn'
        },
        'fraud': {
            'num': 5,
            'name': 'FRAUD DETECTION XGBOOST',
            'script': src_dir / 'train_fraud_detection.py',
            'total': '5.8M samples',
            'malicious': '785K fraudulent transactions',
            'benign': '5M normal transactions',
            'desc': 'XGBoost for transaction pattern classification',
            'category': 'sklearn'
        },
        'host': {
            'num': 6,
            'name': 'HOST BEHAVIOR RF',
            'script': src_dir / 'train_host_behavior.py',
            'total': '5.5M samples',
            'malicious': '500K malware patterns',
            'benign': '5M normal host behavior',
            'desc': 'RandomForest for process/memory pattern classification',
            'category': 'sklearn'
        }
    }
    
    # Determine which models to train
    if args.models == 'all':
        models_to_train = list(models.keys())
    elif args.models == 'pytorch':
        models_to_train = [k for k, v in models.items() if v['category'] == 'pytorch']
    elif args.models == 'sklearn':
        models_to_train = [k for k, v in models.items() if v['category'] == 'sklearn']
    else:
        models_to_train = [m.strip() for m in args.models.split(',')]
    
    # Train models
    results = {}
    start_time = time.time()
    
    for model_key in models_to_train:
        if model_key not in models:
            print(f"Unknown model: {model_key}")
            continue
        
        m = models[model_key]
        
        # Print model info
        print_model_info(m['num'], m['name'], m['total'], m['malicious'], m['benign'], m['desc'])
        
        # Discord notification
        if notifier:
            notifier.model_started(m['name'], m['total'], m['malicious'], m['benign'])
        
        # Train
        success, elapsed = run_training_script(m['script'], m['name'])
        results[model_key] = success
        
        # Discord notification
        if notifier:
            notifier.model_completed(m['name'], success, elapsed)
        
        if success:
            print(f"  ✓ {m['name']} completed in {elapsed}")
        else:
            print(f"  ✗ {m['name']} failed after {elapsed}")
    
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
    for model_key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {models[model_key]['name']:30s}: {status}")
    print("=" * 80)
    
    if notifier:
        notifier.pipeline_completed(str(timedelta(seconds=int(total_time))), successful, failed)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
