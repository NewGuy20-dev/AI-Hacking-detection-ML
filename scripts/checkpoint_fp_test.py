#!/usr/bin/env python3
"""Checkpoint-enabled FP test that can resume from where it left off."""
import json
import pickle
from pathlib import Path
from datetime import datetime

class FPTestCheckpoint:
    def __init__(self, checkpoint_file="fp_test_checkpoint.pkl"):
        self.checkpoint_file = Path(checkpoint_file)
        self.state = {
            'completed_categories': [],
            'current_category': None,
            'current_progress': 0,
            'total_fp': 0,
            'total_tested': 0,
            'category_stats': {},
            'fp_examples': [],
            'start_time': datetime.now().isoformat()
        }
    
    def load_checkpoint(self):
        """Load existing checkpoint if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            print(f"✓ Loaded checkpoint: {len(self.state['completed_categories'])} categories done")
            return True
        return False
    
    def save_checkpoint(self):
        """Save current progress."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.state, f)
        print(f"💾 Checkpoint saved: {len(self.state['completed_categories'])}/{11} categories")
    
    def update_progress(self, category, fp_count, tested_count, fp_rate):
        """Update progress for current category."""
        self.state['category_stats'][category] = (fp_count, tested_count, fp_rate)
        self.state['total_fp'] += fp_count
        self.state['total_tested'] += tested_count
        
        if category not in self.state['completed_categories']:
            self.state['completed_categories'].append(category)
        
        self.save_checkpoint()
    
    def get_remaining_categories(self, all_categories):
        """Get list of categories still to process."""
        return [cat for cat in all_categories if cat[0] not in self.state['completed_categories']]
    
    def cleanup(self):
        """Remove checkpoint file when done."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

# Quick script to check current FP test progress
def check_fp_progress():
    checkpoint = FPTestCheckpoint()
    if checkpoint.load_checkpoint():
        completed = len(checkpoint.state['completed_categories'])
        total_cats = 11  # From test_fp_5m.py
        progress = completed / total_cats * 100
        
        print(f"FP Test Progress: {completed}/{total_cats} categories ({progress:.1f}%)")
        print(f"Completed: {checkpoint.state['completed_categories']}")
        print(f"Total tested so far: {checkpoint.state['total_tested']:,}")
        print(f"Current FP rate: {checkpoint.state['total_fp']/max(checkpoint.state['total_tested'],1)*100:.2f}%")
        
        # Estimate remaining time
        if completed > 0:
            start_time = datetime.fromisoformat(checkpoint.state['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds() / 3600  # hours
            time_per_category = elapsed / completed
            remaining_time = time_per_category * (total_cats - completed)
            print(f"Estimated remaining: {remaining_time:.1f} hours")
    else:
        print("No checkpoint found - FP test not started or completed")

if __name__ == "__main__":
    check_fp_progress()
