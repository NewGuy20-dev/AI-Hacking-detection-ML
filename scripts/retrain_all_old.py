#!/usr/bin/env python3
"""Retrain all models with new generated data."""
import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

class RetrainCheckpoint:
    """Manage retrain_all.py checkpoint state."""
    
    def __init__(self, checkpoint_file="retrain_checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.state = {
            'completed_phases': [],
            'completed_models': [],
            'current_phase': None,
            'start_time': datetime.now().isoformat(),
            'fp_test_progress': {}
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
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f" {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
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
        # Live benign data
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
    parser = argparse.ArgumentParser(description="Retrain all models with resume capability")
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--resume-data', action='store_true', help='Resume from data generation phase')
    parser.add_argument('--resume-pytorch', action='store_true', help='Resume from PyTorch training phase')
    parser.add_argument('--resume-sklearn', action='store_true', help='Resume from sklearn training phase')
    parser.add_argument('--resume-validation', action='store_true', help='Resume from validation phase')
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent
    scripts = base / "scripts"
    training = base / "src" / "training"
    src = base / "src"
    
    # Initialize checkpoint
    checkpoint = RetrainCheckpoint(base / "retrain_checkpoint.json")
    
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
    
    print("=" * 60)
    print(" FULL MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.resume:
        print(f" Resuming from phase {start_phase}")
    print("=" * 60)
    
    # Check/generate data
    if start_phase <= 1 and not checkpoint.is_phase_complete('data_generation'):
        print("\n[Phase 1] Checking/Generating Training Data")
        print("-" * 40)
        
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
    
    # Train PyTorch models
    if start_phase <= 2 and not checkpoint.is_phase_complete('pytorch_training'):
        print("\n[Phase 2] Training PyTorch Models")
        print("-" * 40)
        
        # 1. Payload CNN (~8-9 hours)
        if not checkpoint.is_model_complete('payload_cnn'):
            model_file = base / "models" / "payload_cnn.pt"
            if model_file.exists():
                print("✓ Payload CNN already trained")
                checkpoint.mark_model_complete('payload_cnn')
            else:
                cmd = [sys.executable, str(training / "train_payload.py")]
                if args.resume:
                    cmd.append("--resume")
                subprocess.run(cmd, cwd=training)
                checkpoint.mark_model_complete('payload_cnn')
        
        # 2. URL CNN (~2-3 hours)
        if not checkpoint.is_model_complete('url_cnn'):
            run_script(training / "train_url.py", "URL CNN (10M samples, ~2-3h)")
            checkpoint.mark_model_complete('url_cnn')
        
        # 3. Timeseries LSTM (~30-45 min)
        if not checkpoint.is_model_complete('timeseries_lstm'):
            run_script(training / "train_timeseries.py", "Timeseries LSTM (500k samples, ~30-45m)")
            checkpoint.mark_model_complete('timeseries_lstm')
        
        checkpoint.mark_phase_complete('pytorch_training')
    
    # Train sklearn models
    if start_phase <= 3 and not checkpoint.is_phase_complete('sklearn_training'):
        print("\n[Phase 3] Training Sklearn Models")
        print("-" * 40)
        
        # 4. Network Intrusion
        if not checkpoint.is_model_complete('network_intrusion'):
            run_script(src / "train_network_intrusion.py", "Network Intrusion RF (1M samples, ~15m)")
            checkpoint.mark_model_complete('network_intrusion')
        
        # 5. Fraud Detection
        if not checkpoint.is_model_complete('fraud_detection'):
            run_script(src / "train_fraud_detection.py", "Fraud Detection XGBoost (500k samples, ~10m)")
            checkpoint.mark_model_complete('fraud_detection')
        
        # 6. Host Behavior
        if not checkpoint.is_model_complete('host_behavior'):
            run_script(src / "train_host_behavior.py", "Host Behavior RF (500k samples, ~10m)")
            checkpoint.mark_model_complete('host_behavior')
        
        checkpoint.mark_phase_complete('sklearn_training')
    
    # Validation
    if start_phase <= 4 and not checkpoint.is_phase_complete('validation'):
        print("\n[Phase 4] Validation")
        print("-" * 40)
        
        if not checkpoint.is_model_complete('realworld_validation'):
            run_script(scripts / "validate_realworld.py", "Real-world validation")
            checkpoint.mark_model_complete('realworld_validation')
        
        if not checkpoint.is_model_complete('fp_test_5m'):
            # Create checkpointed FP test if needed
            create_checkpointed_fp_test(base)
            run_script(scripts / "test_fp_5m_checkpointed.py", "FP test (5M samples)")
            checkpoint.mark_model_complete('fp_test_5m')
        
        checkpoint.mark_phase_complete('validation')
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(" RETRAINING COMPLETE")
    print(f" Total time: {timedelta(seconds=int(total_time))}")
    print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Cleanup checkpoint
    checkpoint.cleanup()
    parser = argparse.ArgumentParser(description="Retrain all models with resume capability")
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--resume-data', action='store_true', help='Resume from data generation phase')
    parser.add_argument('--resume-pytorch', action='store_true', help='Resume from PyTorch training phase')
    parser.add_argument('--resume-sklearn', action='store_true', help='Resume from sklearn training phase')
    parser.add_argument('--resume-validation', action='store_true', help='Resume from validation phase')
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent
    scripts = base / "scripts"
    training = base / "src" / "training"
    src = base / "src"
    
    # Initialize checkpoint
    checkpoint = RetrainCheckpoint(base / "retrain_checkpoint.json")
    
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
    
    print("=" * 60)
    print(" FULL MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.resume:
        print(f" Resuming from phase {start_phase}")
    print("=" * 60)
    
    # Phase 1: Data Generation
    if start_phase <= 1 and not checkpoint.is_phase_complete('data_generation'):
        print("\n[Phase 1] Checking/Generating Training Data")
        print("-" * 40)
        
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
        print("-" * 40)
        
        pytorch_models = [
            (training / "train_payload.py", "Payload CNN", "payload_cnn"),
            (training / "train_url.py", "URL CNN", "url_cnn"), 
            (training / "train_timeseries.py", "Timeseries LSTM", "timeseries_lstm")
        ]
        
        for script, desc, model_name in pytorch_models:
            if not checkpoint.is_model_complete(model_name):
                # Check if model file exists (training completed)
                model_file = base / "models" / f"{model_name}.pt"
                if model_file.exists():
                    print(f"✓ {desc} already trained")
                    checkpoint.mark_model_complete(model_name)
                else:
                    # Add --resume flag for PyTorch training
                    cmd = [sys.executable, str(script)]
                    if args.resume or checkpoint.is_model_complete(f"{model_name}_started"):
                        cmd.append("--resume")
                    
                    print(f"Training {desc}...")
                    checkpoint.mark_model_complete(f"{model_name}_started")
                    
                    result = subprocess.run(cmd, cwd=script.parent)
                    if result.returncode == 0:
                        checkpoint.mark_model_complete(model_name)
        
        checkpoint.mark_phase_complete('pytorch_training')
    
    # Phase 3: Sklearn Models  
    if start_phase <= 3 and not checkpoint.is_phase_complete('sklearn_training'):
        print("\n[Phase 3] Training Sklearn Models")
        print("-" * 40)
        
        sklearn_models = [
            (src / "train_network_intrusion.py", "Network Intrusion RF", "network_intrusion"),
            (src / "train_fraud_detection.py", "Fraud Detection XGBoost", "fraud_detection"),
            (src / "train_host_behavior.py", "Host Behavior RF", "host_behavior")
        ]
        
        for script, desc, model_name in sklearn_models:
            if not checkpoint.is_model_complete(model_name):
                model_file = base / "models" / f"{model_name}_model.pkl"
                if model_file.exists():
                    print(f"✓ {desc} already trained")
                    checkpoint.mark_model_complete(model_name)
                else:
                    run_script(script, desc)
                    checkpoint.mark_model_complete(model_name)
        
        checkpoint.mark_phase_complete('sklearn_training')
    
    # Phase 4: Validation
    if start_phase <= 4 and not checkpoint.is_phase_complete('validation'):
        print("\n[Phase 4] Validation")
        print("-" * 40)
        
        validation_scripts = [
            (scripts / "validate_realworld.py", "Real-world validation", "realworld_validation"),
            (scripts / "test_fp_5m.py", "FP test (5M samples)", "fp_test_5m")
        ]
        
        for script, desc, test_name in validation_scripts:
            if not checkpoint.is_model_complete(test_name):
                # For FP test, add checkpoint capability
                if "test_fp_5m" in str(script):
                    # Create checkpointed version
                    create_checkpointed_fp_test(base)
                    script = base / "scripts" / "test_fp_5m_checkpointed.py"
                
                run_script(script, desc)
                checkpoint.mark_model_complete(test_name)
        
        checkpoint.mark_phase_complete('validation')
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(" RETRAINING COMPLETE")
    print(f" Total time: {timedelta(seconds=int(total_time))}")
    print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Cleanup checkpoint
    checkpoint.cleanup()


def create_checkpointed_fp_test(base_path):
    """Create a checkpointed version of FP test."""
    original = base_path / "scripts" / "test_fp_5m.py"
    checkpointed = base_path / "scripts" / "test_fp_5m_checkpointed.py"
    
    if checkpointed.exists():
        return
    
    # Read original and add checkpoint functionality
    with open(original, 'r') as f:
        content = f.read()
    
    # Add checkpoint imports and class
    checkpoint_code = '''
import pickle

class FPTestCheckpoint:
    def __init__(self, checkpoint_file="fp_test_checkpoint.pkl"):
        self.checkpoint_file = Path(checkpoint_file)
        self.state = {
            'completed_categories': [],
            'total_fp': 0,
            'total_tested': 0,
            'category_stats': {},
            'fp_examples': []
        }
    
    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            return True
        return False
    
    def save_checkpoint(self):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.state, f)
    
    def is_category_done(self, category):
        return category in self.state['completed_categories']
    
    def mark_category_done(self, category, fp, tested, rate):
        self.state['completed_categories'].append(category)
        self.state['category_stats'][category] = (fp, tested, rate)
        self.state['total_fp'] += fp
        self.state['total_tested'] += tested
        self.save_checkpoint()
'''
    
    # Insert checkpoint code after imports
    import_end = content.find('def gen_sentences')
    modified_content = content[:import_end] + checkpoint_code + '\n\n' + content[import_end:]
    
    # Modify test_fp function to use checkpoints
    modified_content = modified_content.replace(
        'def test_fp():',
        '''def test_fp():
    checkpoint = FPTestCheckpoint()
    checkpoint.load_checkpoint()'''
    )
    
    # Add checkpoint logic in the main loop (simplified version)
    modified_content = modified_content.replace(
        'for cat_name, gen_func, count in generators:',
        '''for cat_name, gen_func, count in generators:
        if checkpoint.is_category_done(cat_name):
            print(f"✓ Skipping {cat_name} (already completed)")
            continue'''
    )
    
    with open(checkpointed, 'w') as f:
        f.write(modified_content)




if __name__ == "__main__":
    main()
