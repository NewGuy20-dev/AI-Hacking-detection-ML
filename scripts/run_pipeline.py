#!/usr/bin/env python3
"""Run complete pipeline: data generation, baseline, holdout, and validation."""
import subprocess
import sys
from pathlib import Path


def run_script(script_path, desc):
    """Run a script and return success status."""
    print(f"\n{'='*60}")
    print(f" {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent.parent)
    if result.returncode != 0:
        print(f"  ✗ {desc} failed")
        return False
    print(f"  ✓ {desc} complete")
    return True


def main():
    base = Path(__file__).parent.parent
    scripts = base / "scripts"
    
    print("="*60)
    print(" COMPLETE PIPELINE EXECUTION")
    print("="*60)
    
    steps = [
        (scripts / "generate_benign_data.py", "Generate curated benign data"),
        (scripts / "generate_adversarial_benign.py", "Generate adversarial benign data"),
        (scripts / "generate_500k_benign_test.py", "Generate 500k FP test dataset"),
        (scripts / "create_holdout_set.py", "Create holdout test set"),
        (scripts / "establish_baseline.py", "Establish baseline metrics"),
        (scripts / "retrain_all.py", "Retrain all models"),
        (scripts / "validate_metrics.py", "Validate against targets"),
    ]
    
    results = []
    for script, desc in steps:
        if script.exists():
            success = run_script(script, desc)
            results.append((desc, success))
        else:
            print(f"\n  ⚠ Skipping {desc} - script not found")
            results.append((desc, None))
    
    # Summary
    print(f"\n{'='*60}")
    print(" PIPELINE SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        if success is None:
            status = "⚠ SKIPPED"
        elif success:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {status}: {desc}")
    
    passed = sum(1 for _, s in results if s is True)
    total = sum(1 for _, s in results if s is not None)
    print(f"\nResult: {passed}/{total} steps passed")


if __name__ == "__main__":
    main()
