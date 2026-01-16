"""Tier 4: FP Testing - Test false positive rate on benign samples."""
import sys
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN

FP_TARGET = 0.02  # 2% max false positive rate


def load_benign_samples(data_dir: Path, max_samples: int = 500000):
    """Load benign samples from fp_test_500k.jsonl."""
    fp_file = data_dir / 'fp_test_500k.jsonl'
    if not fp_file.exists():
        return None
    
    samples = []
    with open(fp_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                samples.append(item.get('text', item.get('payload', '')))
            except:
                continue
    
    return samples


def encode_batch(texts, max_len=500, vocab_size=256):
    """Encode batch of texts."""
    encoded = []
    for text in texts:
        seq = [min(ord(c), vocab_size - 1) for c in text[:max_len]]
        seq = seq + [0] * (max_len - len(seq))
        encoded.append(seq)
    return np.array(encoded, dtype=np.int64)


def run_fp_test(models_dir: Path, data_dir: Path, device: str = 'cpu', max_samples: int = 500000) -> dict:
    """Run false positive test on benign samples."""
    results = {'passed': True, 'samples': 0, 'fp_rate': {}, 'target': FP_TARGET}
    
    print("\n" + "="*60)
    print("TIER 4: FALSE POSITIVE TESTING")
    print("="*60)
    
    # Load benign samples
    samples = load_benign_samples(data_dir, max_samples)
    if samples is None:
        print("  ⚠ FP test data not found (fp_test_500k.jsonl)")
        results['passed'] = False
        return results
    
    results['samples'] = len(samples)
    print(f"  Loaded {len(samples):,} benign samples")
    
    # Test PyTorch models
    pytorch_configs = [
        ('payload', PayloadCNN, 'payload_cnn_best.pth', 500, 256),
        ('url', URLCNN, 'url_cnn_best.pth', 200, 128),
    ]
    
    batch_size = 1024
    
    for name, cls, filename, max_len, vocab_size in pytorch_configs:
        path = models_dir / filename
        if not path.exists():
            results['fp_rate'][name] = None
            print(f"  {name:15s}: SKIP (not found)")
            continue
        
        try:
            model = cls()
            state = torch.load(path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            model.to(device).eval()
            
            # Process in batches
            false_positives = 0
            total = 0
            
            for i in tqdm(range(0, len(samples), batch_size), desc=f"  {name}", leave=False):
                batch = samples[i:i+batch_size]
                X = encode_batch(batch, max_len, vocab_size)
                X = torch.tensor(X).to(device)
                
                with torch.no_grad():
                    logits = model(X)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                false_positives += (probs >= 0.5).sum()
                total += len(batch)
            
            fp_rate = false_positives / total
            results['fp_rate'][name] = fp_rate
            
            passed = fp_rate <= FP_TARGET
            status = "✓" if passed else "⚠"
            print(f"  {name:15s}: {status} FP rate = {fp_rate:.4f} ({false_positives:,}/{total:,})")
            
            if not passed:
                results['passed'] = False
                
        except Exception as e:
            results['fp_rate'][name] = None
            print(f"  {name:15s}: ✗ {e}")
    
    print("-"*60)
    overall = "PASSED" if results['passed'] else "WARNING"
    print(f"FP Testing: {overall} (target: <{FP_TARGET*100:.0f}%)")
    
    return results
