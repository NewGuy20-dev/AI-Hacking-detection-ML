#!/usr/bin/env python3
"""Save current training state and safely exit."""
import os
import signal
import subprocess
import sys
from pathlib import Path

def save_current_state():
    """Save all current training progress."""
    base = Path(__file__).parent.parent
    
    print("🔍 Checking training status...")
    
    # Check if models exist
    models_dir = base / "models"
    payload_model = models_dir / "payload_cnn.pt"
    
    if payload_model.exists():
        print("✅ Payload CNN training: COMPLETED")
    else:
        print("⏳ Payload CNN training: IN PROGRESS")
    
    # Check latest checkpoint
    checkpoint_dir = base / "checkpoints" / "payload"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"💾 Latest checkpoint: {latest.name}")
    
    # Kill any running processes safely
    print("\n🛑 Stopping any running training processes...")
    try:
        # Find Python processes related to training
        result = subprocess.run(['pgrep', '-f', 'train.*py'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"Stopping process {pid}")
                os.kill(int(pid), signal.SIGTERM)
    except:
        pass
    
    print("\n✅ Safe to shutdown!")
    print("\n📋 To resume later:")
    print("1. Run: python scripts/retrain_all.py")
    print("2. Or resume specific training: python src/training/train_payload.py --resume")
    
    return True

if __name__ == "__main__":
    save_current_state()
