#!/usr/bin/env python3
"""Test payload model FP rate on 5M fresh benign samples with checkpoint support."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random
import string
import pickle
import json

# Import all generators from original test
exec(open(Path(__file__).parent / "test_fp_5m.py").read().split('def load_model')[0])

class FPTestCheckpoint:
    """Checkpoint system for FP testing."""
    
    def __init__(self, checkpoint_file="fp_test_checkpoint.pkl"):
        self.checkpoint_file = Path(checkpoint_file)
        self.state = {
            'completed_categories': [],
            'total_fp': 0,
            'total_tested': 0,
            'category_stats': {},
            'fp_examples': [],
            'start_time': datetime.now().isoformat()
        }
    
    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            print(f"✓ Loaded checkpoint: {len(self.state['completed_categories'])}/11 categories done")
            return True
        return False
    
    def save_checkpoint(self):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.state, f)
        completed = len(self.state['completed_categories'])
        print(f"💾 Checkpoint saved: {completed}/11 categories ({completed/11*100:.1f}%)")
    
    def is_category_done(self, category):
        return category in self.state['completed_categories']
    
    def mark_category_done(self, category, fp, tested, rate, examples):
        self.state['completed_categories'].append(category)
        self.state['category_stats'][category] = (fp, tested, rate)
        self.state['total_fp'] += fp
        self.state['total_tested'] += tested
        self.state['fp_examples'].extend(examples)
        self.save_checkpoint()
    
    def get_progress(self):
        completed = len(self.state['completed_categories'])
        if completed > 0:
            start_time = datetime.fromisoformat(self.state['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            time_per_category = elapsed / completed
            remaining = time_per_category * (11 - completed)
            return completed, 11, elapsed, remaining
        return 0, 11, 0, 0
    
    def cleanup(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


def load_model(model_path):
    """Load payload CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try TorchScript first, then regular checkpoint
    try:
        model = torch.jit.load(model_path, map_location=device)
    except:
        from torch_models.payload_cnn import PayloadCNN
        model = PayloadCNN(vocab_size=256, embed_dim=128, num_filters=256, max_len=500)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        model.to(device)
    
    model.eval()
    return model, device


def encode_batch(texts, max_len=500):
    """Encode texts to tensor."""
    batch = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        encoded = [ord(c) % 256 for c in text[:max_len]]
        batch[i, :len(encoded)] = encoded
    return torch.tensor(batch)


def test_fp_checkpointed():
    base = Path(__file__).parent.parent
    model_path = base / "models" / "payload_cnn.pt"
    
    print("=" * 60)
    print(" FALSE POSITIVE TEST - 5M FRESH BENIGN SAMPLES (CHECKPOINTED)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_path}")
    
    # Initialize checkpoint
    checkpoint = FPTestCheckpoint(base / "fp_test_checkpoint.pkl")
    checkpoint.load_checkpoint()
    
    # Show progress
    completed, total, elapsed, remaining = checkpoint.get_progress()
    if completed > 0:
        print(f"Progress: {completed}/{total} categories ({completed/total*100:.1f}%)")
        print(f"Elapsed: {elapsed:.1f}h, Estimated remaining: {remaining:.1f}h")
    
    # Load model
    print("\n--- Loading Model ---")
    model, device = load_model(model_path)
    print(f"Device: {device}")
    
    # Generators (500k each = 5M total)
    generators = [
        ("Sentences", gen_sentences, 500_000),
        ("Emails", gen_emails, 500_000),
        ("File Paths", gen_paths, 500_000),
        ("Products", gen_products, 500_000),
        ("Log Entries", gen_logs, 500_000),
        ("JSON", gen_json, 500_000),
        ("URLs", gen_urls, 500_000),
        ("Code Snippets", gen_code, 500_000),
        ("Names", gen_names, 500_000),
        ("Math", gen_math, 250_000),
        ("Addresses", gen_addresses, 250_000),
    ]
    
    batch_size = 2048
    
    with torch.no_grad():
        for cat_name, gen_func, count in generators:
            # Skip if already completed
            if checkpoint.is_category_done(cat_name):
                print(f"✓ Skipping {cat_name} (already completed)")
                continue
            
            print(f"\n--- Processing {cat_name} ({count:,} samples) ---")
            
            cat_fp = 0
            cat_tested = 0
            batch = []
            cat_examples = []
            
            pbar = tqdm(gen_func(count), total=count, desc=cat_name)
            for text in pbar:
                batch.append(text)
                
                if len(batch) >= batch_size:
                    inputs = encode_batch(batch).to(device)
                    outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                    
                    fps = outputs > 0.5
                    cat_fp += fps.sum()
                    cat_tested += len(batch)
                    
                    # Save FP examples
                    if fps.any() and len(cat_examples) < 20:
                        for j, (is_fp, score) in enumerate(zip(fps, outputs)):
                            if is_fp and len(cat_examples) < 20:
                                cat_examples.append((cat_name, batch[j][:80], score))
                    
                    batch = []
                    pbar.set_postfix(fp_rate=f"{cat_fp/max(cat_tested,1)*100:.3f}%")
            
            # Process remaining
            if batch:
                inputs = encode_batch(batch).to(device)
                outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                fps = outputs > 0.5
                cat_fp += fps.sum()
                cat_tested += len(batch)
            
            fp_rate = cat_fp / max(cat_tested, 1) * 100
            print(f"  {cat_name}: {cat_fp:,}/{cat_tested:,} FP ({fp_rate:.3f}%)")
            
            # Mark category as done
            checkpoint.mark_category_done(cat_name, cat_fp, cat_tested, fp_rate, cat_examples)
    
    # Final results
    overall_fp_rate = checkpoint.state['total_fp'] / checkpoint.state['total_tested'] * 100
    
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)
    print(f"Total Samples: {checkpoint.state['total_tested']:,}")
    print(f"False Positives: {checkpoint.state['total_fp']:,}")
    print(f"FP Rate: {overall_fp_rate:.4f}%")
    print(f"Target: <2-3%")
    print(f"Status: {'✓ PASS' if overall_fp_rate < 3 else '✗ FAIL'}")
    
    # Save final results
    results_file = base / "evaluation" / f"fp_test_5m_checkpointed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        f.write(f"FP Test Results (Checkpointed) - {datetime.now().isoformat()}\n")
        f.write(f"Total: {checkpoint.state['total_tested']:,} samples\n")
        f.write(f"FP: {checkpoint.state['total_fp']:,} ({overall_fp_rate:.4f}%)\n\n")
        for cat, (fp, tested, rate) in checkpoint.state['category_stats'].items():
            f.write(f"{cat}: {fp}/{tested} ({rate:.3f}%)\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Cleanup checkpoint
    checkpoint.cleanup()
    
    return overall_fp_rate < 3


if __name__ == "__main__":
    success = test_fp_checkpointed()
    sys.exit(0 if success else 1)
