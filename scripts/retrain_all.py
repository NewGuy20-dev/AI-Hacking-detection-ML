"""Retrain all PyTorch models with improved data."""
import subprocess
import sys
import argparse
from pathlib import Path

def run_script(script_path, desc):
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent.parent)
    return result.returncode == 0

def _run_feedback_loop(base: Path, run_date: str = None, promote: bool = False):
    """Optional hard-example feedback loop integration."""
    loop_script = base / 'src' / 'feedback_loop' / 'hard_example_loop.py'
    if not loop_script.exists():
        print("Feedback loop script not found, skipping.")
        return

    cmd = [sys.executable, str(loop_script), '--model', 'payload,url']
    if run_date:
        cmd.extend(['--run-date', run_date])
    if promote:
        cmd.append('--promote')

    print("\n" + "=" * 60)
    print(" FEEDBACK LOOP (PAYLOAD+URL)")
    print("=" * 60)
    subprocess.run(cmd, cwd=base)


def main():
    parser = argparse.ArgumentParser(description="Full retraining pipeline")
    parser.add_argument('--with-feedback-loop', action='store_true',
                        help='Run hard-example feedback loop before retraining payload/url')
    parser.add_argument('--feedback-run-date', type=str, default=None,
                        help='Optional run date (YYYY-MM-DD) for selecting stress failure logs')
    parser.add_argument('--feedback-promote', action='store_true',
                        help='Allow feedback loop model promotion if strict gates pass')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    training_dir = base / 'src' / 'training'
    scripts_dir = base / 'scripts'
    
    print("="*60)
    print(" FULL MODEL RETRAINING PIPELINE")
    print("="*60)

    if args.with_feedback_loop:
        _run_feedback_loop(base, run_date=args.feedback_run_date, promote=args.feedback_promote)
    
    # Step 1: Generate fresh benign data
    print("\n[1/7] Generating curated benign data...")
    run_script(scripts_dir / 'generate_benign_data.py', "Benign Data Generation")
    
    # Step 2: Generate ADVERSARIAL benign data (critical for reducing false positives)
    print("\n[2/7] Generating adversarial benign data...")
    run_script(scripts_dir / 'generate_adversarial_benign.py', "Adversarial Benign Data Generation")
    
    # Step 3: Generate 500k FP test dataset
    print("\n[3/7] Generating 500k FP test dataset...")
    run_script(scripts_dir / 'generate_500k_benign_test.py', "500k Benign FP Test Generation")
    
    # Step 4: Generate improved synthetic URLs  
    print("\n[4/7] Generating improved URL data...")
    run_script(scripts_dir / 'generate_improved_urls.py', "Improved URL Generation")
    
    # Step 5: Retrain Payload CNN
    print("\n[5/7] Retraining Payload CNN...")
    run_script(training_dir / 'train_payload.py', "Payload CNN Training")
    
    # Step 6: Retrain URL CNN
    print("\n[6/7] Retraining URL CNN...")
    run_script(training_dir / 'train_url.py', "URL CNN Training")
    
    # Step 7: Retrain Time-Series LSTM
    print("\n[7/7] Retraining Time-Series LSTM...")
    run_script(training_dir / 'train_timeseries.py', "Time-Series LSTM Training")
    
    # Validation
    print("\n" + "="*60)
    print(" Running Validation...")
    print("="*60)
    run_script(scripts_dir / 'validate_realworld.py', "Real-World Validation")
    
    print("\n✓ Retraining complete!")

if __name__ == "__main__":
    main()
