#!/usr/bin/env python3
"""Retrain all models with specialized data and checkpoint support."""
import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

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
    
    def pipeline_started(self, total_models: int, total_samples: str):
        """🚀 Pipeline started."""
        embed = {
            "title": "🚀 Model Retraining Pipeline Started",
            "color": self.COLORS['info'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Models", "value": str(total_models), "inline": True},
                {"name": "Total Samples", "value": total_samples, "inline": True},
                {"name": "Data Ratio", "value": "67% Benign : 33% Malicious", "inline": True},
            ]
        }
        return self._send(embed)
    
    def model_started(self, model_name: str, samples: str, description: str):
        """📦 Model training started."""
        embed = {
            "title": f"📦 Training: {model_name}",
            "color": self.COLORS['info'],
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Samples", "value": samples, "inline": True},
            ]
        }
        return self._send(embed)
    
    def model_completed(self, model_name: str, success: bool, elapsed: str):
        """✅ Model training completed."""
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
            "title": "🎉 Retraining Pipeline Complete!",
            "color": self.COLORS['success'] if failed == 0 else self.COLORS['warning'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Total Time", "value": total_time, "inline": True},
                {"name": "Successful", "value": str(successful), "inline": True},
                {"name": "Failed", "value": str(failed), "inline": True},
            ]
        }
        return self._send(embed)


class RetrainCheckpoint:
    """Manage checkpoint state."""
    
    def __init__(self, checkpoint_file="retrain_checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.state = {
            'completed_phases': [],
            'completed_models': [],
            'current_phase': None,
            'start_time': datetime.now().isoformat(),
        }
    
    def load_checkpoint(self):
        """Load existing checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
            return True
        return False
    
    def save_checkpoint(self):
        """Save current state."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_phase_complete(self, phase):
        """Mark a phase as completed."""
        if phase not in self.state['completed_phases']:
            self.state['completed_phases'].append(phase)
        self.save_checkpoint()
    
    def mark_model_complete(self, model):
        """Mark a model as completed."""
        if model not in self.state['completed_models']:
            self.state['completed_models'].append(model)
        self.save_checkpoint()
    
    def is_phase_complete(self, phase):
        """Check if phase is already completed."""
        return phase in self.state['completed_phases']
    
    def is_model_complete(self, model):
        """Check if model is already completed."""
        return model in self.state['completed_models']
    
    def cleanup(self):
        """Remove checkpoint when fully complete."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


def run_script(script_path, desc, timeout=None):
    """Run a script and return success status."""
    print(f"\n{'='*80}")
    print(f" {desc}")
    print(f" {datetime.now().strftime('%H:%M:%S')}")
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
        print(f"  Completed in {timedelta(seconds=int(elapsed))}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_data_exists(base):
    """Check if generated data exists."""
    files = [
        ("URL malicious 5M", base / "datasets/url_analysis/malicious_urls_5m.jsonl"),
        ("Network intrusion 500k", base / "datasets/network_intrusion/synthetic_500k.jsonl"),
        ("Fraud 500k", base / "datasets/fraud_detection/synthetic_500k.jsonl"),
        ("Timeseries 500k", base / "datasets/timeseries/normal_traffic_500k.npy"),
        ("Host behavior 500k", base / "datasets/host_behavior/synthetic_500k.jsonl"),
        ("Wikipedia text", base / "datasets/live_benign/wikipedia_text.jsonl"),
        ("GitHub snippets", base / "datasets/live_benign/github_snippets.jsonl"),
        ("Fraud benign", base / "datasets/live_benign/fraud_benign.jsonl"),
        ("Host behavior benign", base / "datasets/live_benign/host_behavior_benign.jsonl"),
        ("Timeseries benign", base / "datasets/live_benign/timeseries_benign.npy"),
    ]
    
    missing = []
    for name, path in files:
        if not path.exists():
            missing.append((name, path))
    
    return missing


def main():
    parser = argparse.ArgumentParser(description="Retrain all models with specialized data")
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--resume-data', action='store_true', help='Resume from data generation phase')
    parser.add_argument('--resume-pytorch', action='store_true', help='Resume from PyTorch training phase')
    parser.add_argument('--resume-sklearn', action='store_true', help='Resume from sklearn training phase')
    parser.add_argument('--resume-validation', action='store_true', help='Resume from validation phase')
    parser.add_argument('--no-discord', action='store_true', help='Disable Discord notifications')
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent
    scripts = base / "scripts"
    training = base / "src" / "training"
    src = base / "src"
    
    # Create checkpoints directory
    checkpoint_dir = base / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize Discord notifier
    notifier = DiscordNotifier() if not args.no_discord else None
    
    # Initialize checkpoint
    checkpoint = RetrainCheckpoint(checkpoint_dir / "retrain_checkpoint.json")
    
    # Load existing checkpoint if resuming
    if args.resume or any([args.resume_data, args.resume_pytorch, args.resume_sklearn, args.resume_validation]):
        if checkpoint.load_checkpoint():
            print(f"✓ Loaded checkpoint - completed phases: {checkpoint.state['completed_phases']}")
        else:
            print("No checkpoint found, starting fresh")
    
    # Determine starting phase
    start_phase = 1
    if args.resume_pytorch or 'data_generation' in checkpoint.state['completed_phases']:
        start_phase = 2
    elif args.resume_sklearn or 'pytorch_training' in checkpoint.state['completed_phases']:
        start_phase = 3
    elif args.resume_validation or 'sklearn_training' in checkpoint.state['completed_phases']:
        start_phase = 4
    
    start_time = time.time()
    
    print("=" * 80)
    print(" SPECIALIZED MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.resume:
        print(f" Resuming from phase {start_phase}")
    print("=" * 80)
    print("\n📊 DATASET OVERVIEW")
    print("  Total: 322.6M samples (54.51 GB)")
    print("  Benign: 215.6M (66.8%)")
    print("  Malicious: 107.1M (33.2%)")
    print("  Ratio: 2:1 (optimal for production)")
    print("=" * 80)
    
    if notifier:
        notifier.pipeline_started(6, "322.6M")
    
    # Phase 1: Data Generation
    if start_phase <= 1 and not checkpoint.is_phase_complete('data_generation'):
        print("\n[Phase 1] Checking/Generating Training Data")
        print("-" * 80)
        
        missing = check_data_exists(base)
        if missing:
            print(f"Missing {len(missing)} datasets, generating...")
            
            data_scripts = [
                (scripts / "generate_malicious_urls_5m.py", "URL malicious 5M"),
                (scripts / "generate_network_intrusion_500k.py", "Network intrusion 500k"),
                (scripts / "generate_fraud_500k.py", "Fraud 500k"),
                (scripts / "generate_timeseries_500k.py", "Timeseries 500k"),
                (scripts / "generate_host_behavior_500k.py", "Host behavior 500k"),
            ]
            
            for script, desc in data_scripts:
                if any(desc in m[0] for m in missing):
                    run_script(script, f"Generating {desc}", timeout=3600)
        else:
            print("✓ All training data exists")
        
        checkpoint.mark_phase_complete('data_generation')
    
    # Phase 2: PyTorch Models
    if start_phase <= 2 and not checkpoint.is_phase_complete('pytorch_training'):
        print("\n[Phase 2] Training PyTorch Models")
        print("-" * 80)
        
        # 1. Payload CNN
        if not checkpoint.is_model_complete('payload_cnn'):
            print("\n[1/6] PAYLOAD CNN - Character-level injection detection")
            print("  Malicious: 94.5M security payloads (SQL, XSS, command injection)")
            print("  Benign: 192M curated + live benign")
            print("  Total: ~286M samples")
            
            if notifier:
                notifier.model_started("Payload CNN", "286M", "Character-level injection detection")
            
            model_file = base / "models" / "payload_cnn.pt"
            if model_file.exists():
                print("✓ Payload CNN already trained")
                checkpoint.mark_model_complete('payload_cnn')
            else:
                start = time.time()
                cmd = [sys.executable, str(training / "train_payload.py")]
                if args.resume:
                    cmd.append("--resume")
                result = subprocess.run(cmd, cwd=training)
                elapsed = timedelta(seconds=int(time.time() - start))
                
                success = result.returncode == 0
                if success:
                    checkpoint.mark_model_complete('payload_cnn')
                if notifier:
                    notifier.model_completed("Payload CNN", success, str(elapsed))
        
        # 2. URL CNN
        if not checkpoint.is_model_complete('url_cnn'):
            print("\n[2/6] URL CNN - Malicious URL detection")
            print("  Malicious: 5.2M malicious URLs (phishing, malware)")
            print("  Benign: 11M benign URLs (Tranco + Common Crawl)")
            print("  Total: 16.2M samples")
            
            if notifier:
                notifier.model_started("URL CNN", "16.2M", "Character-level URL classification")
            
            start = time.time()
            success = run_script(training / "train_url.py", "URL CNN", timeout=10800)
            elapsed = timedelta(seconds=int(time.time() - start))
            
            if success:
                checkpoint.mark_model_complete('url_cnn')
            if notifier:
                notifier.model_completed("URL CNN", success, str(elapsed))
        
        # 3. Timeseries LSTM
        if not checkpoint.is_model_complete('timeseries_lstm'):
            print("\n[3/6] TIMESERIES LSTM - Temporal anomaly detection")
            print("  Malicious: 250K attack sequences (DDoS, portscan)")
            print("  Benign: 1.25M normal traffic sequences")
            print("  Total: 1.5M samples")
            
            if notifier:
                notifier.model_started("Timeseries LSTM", "1.5M", "Temporal sequence classifier")
            
            start = time.time()
            success = run_script(training / "train_timeseries.py", "Timeseries LSTM", timeout=3600)
            elapsed = timedelta(seconds=int(time.time() - start))
            
            if success:
                checkpoint.mark_model_complete('timeseries_lstm')
            if notifier:
                notifier.model_completed("Timeseries LSTM", success, str(elapsed))
        
        checkpoint.mark_phase_complete('pytorch_training')
    
    # Phase 3: Sklearn Models
    if start_phase <= 3 and not checkpoint.is_phase_complete('sklearn_training'):
        print("\n[Phase 3] Training Sklearn Models")
        print("-" * 80)
        
        # 4. Network Intrusion
        if not checkpoint.is_model_complete('network_intrusion'):
            print("\n[4/6] NETWORK INTRUSION - Traffic anomaly detection")
            print("  Malicious: 5.5M attack samples (DoS, Probe, R2L, U2R)")
            print("  Benign: 1M MAWI normal traffic")
            print("  Total: 6.5M samples")
            
            if notifier:
                notifier.model_started("Network Intrusion RF", "6.5M", "41-feature traffic classifier")
            
            start = time.time()
            success = run_script(src / "train_network_intrusion.py", "Network Intrusion RF", timeout=3600)
            elapsed = timedelta(seconds=int(time.time() - start))
            
            if success:
                checkpoint.mark_model_complete('network_intrusion')
            if notifier:
                notifier.model_completed("Network Intrusion RF", success, str(elapsed))
        
        # 5. Fraud Detection
        if not checkpoint.is_model_complete('fraud_detection'):
            print("\n[5/6] FRAUD DETECTION - Transaction anomaly detection")
            print("  Malicious: 785K fraudulent transactions")
            print("  Benign: 5M normal transactions")
            print("  Total: 5.8M samples")
            
            if notifier:
                notifier.model_started("Fraud Detection XGBoost", "5.8M", "Transaction pattern classifier")
            
            start = time.time()
            success = run_script(src / "train_fraud_detection.py", "Fraud Detection XGBoost", timeout=1800)
            elapsed = timedelta(seconds=int(time.time() - start))
            
            if success:
                checkpoint.mark_model_complete('fraud_detection')
            if notifier:
                notifier.model_completed("Fraud Detection XGBoost", success, str(elapsed))
        
        # 6. Host Behavior
        if not checkpoint.is_model_complete('host_behavior'):
            print("\n[6/6] HOST BEHAVIOR - Malware pattern detection")
            print("  Malicious: 500K malware patterns")
            print("  Benign: 5M normal host behavior")
            print("  Total: 5.5M samples")
            
            if notifier:
                notifier.model_started("Host Behavior RF", "5.5M", "Process/memory pattern classifier")
            
            start = time.time()
            success = run_script(src / "train_host_behavior.py", "Host Behavior RF", timeout=1800)
            elapsed = timedelta(seconds=int(time.time() - start))
            
            if success:
                checkpoint.mark_model_complete('host_behavior')
            if notifier:
                notifier.model_completed("Host Behavior RF", success, str(elapsed))
        
        checkpoint.mark_phase_complete('sklearn_training')
    
    # Phase 4: Validation
    if start_phase <= 4 and not checkpoint.is_phase_complete('validation'):
        print("\n[Phase 4] Validation")
        print("-" * 80)
        
        if not checkpoint.is_model_complete('realworld_validation'):
            run_script(scripts / "validate_realworld.py", "Real-world validation")
            checkpoint.mark_model_complete('realworld_validation')
        
        checkpoint.mark_phase_complete('validation')
    
    # Summary
    total_time = time.time() - start_time
    completed = checkpoint.state['completed_models']
    successful = len([m for m in completed if m in ['payload_cnn', 'url_cnn', 'timeseries_lstm', 
                                                      'network_intrusion', 'fraud_detection', 'host_behavior']])
    failed = 6 - successful
    
    print("\n" + "=" * 80)
    print(" RETRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Successful: {successful}/6")
    print(f"Failed: {failed}/6")
    print("\nModel Status:")
    for model in ['payload_cnn', 'url_cnn', 'timeseries_lstm', 'network_intrusion', 'fraud_detection', 'host_behavior']:
        status = "✓" if model in completed else "✗"
        print(f"  {status} {model}")
    print("=" * 80)
    
    if notifier:
        notifier.pipeline_completed(str(timedelta(seconds=int(total_time))), successful, failed)
    
    # Cleanup checkpoint
    if failed == 0:
        checkpoint.cleanup()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
