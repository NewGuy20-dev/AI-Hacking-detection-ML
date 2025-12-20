"""Retrain all PyTorch models with improved data."""
import subprocess
import sys
from pathlib import Path

def run_script(script_path, desc):
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent.parent)
    return result.returncode == 0

def main():
    base = Path(__file__).parent.parent
    training_dir = base / 'src' / 'training'
    scripts_dir = base / 'scripts'
    
    print("="*60)
    print(" FULL MODEL RETRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Generate fresh benign data
    print("\n[1/6] Generating curated benign data...")
    run_script(scripts_dir / 'generate_benign_data.py', "Benign Data Generation")
    
    # Step 2: Generate ADVERSARIAL benign data (critical for reducing false positives)
    print("\n[2/6] Generating adversarial benign data...")
    run_script(scripts_dir / 'generate_adversarial_benign.py', "Adversarial Benign Data Generation")
    
    # Step 3: Generate improved synthetic URLs  
    print("\n[3/6] Generating improved URL data...")
    run_script(scripts_dir / 'generate_improved_urls.py', "Improved URL Generation")
    
    # Step 4: Retrain Payload CNN
    print("\n[4/6] Retraining Payload CNN...")
    run_script(training_dir / 'train_payload.py', "Payload CNN Training")
    
    # Step 5: Retrain URL CNN
    print("\n[5/6] Retraining URL CNN...")
    run_script(training_dir / 'train_url.py', "URL CNN Training")
    
    # Step 6: Retrain Time-Series LSTM
    print("\n[6/6] Retraining Time-Series LSTM...")
    run_script(training_dir / 'train_timeseries.py', "Time-Series LSTM Training")
    
    # Validation
    print("\n" + "="*60)
    print(" Running Validation...")
    print("="*60)
    run_script(scripts_dir / 'validate_realworld.py', "Real-World Validation")
    
    print("\n✓ Retraining complete!")

if __name__ == "__main__":
    main()
