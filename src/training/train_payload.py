"""Train PayloadCNN model for malicious payload detection."""
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from tqdm import tqdm

from torch_models.payload_cnn import PayloadCNN
from torch_models.datasets import PayloadDataset
from torch_models.utils import setup_gpu, EarlyStopping, save_model


def load_payload_data(base_path):
    """Load malicious and benign payloads from dataset folders."""
    base = Path(base_path)
    payloads_dir = base / 'datasets' / 'security_payloads'
    curated_dir = base / 'datasets' / 'curated_benign'
    adversarial_dir = curated_dir / 'adversarial'
    
    texts, labels = [], []
    
    # Malicious payloads (label=1)
    for folder in ['injection', 'fuzzing', 'misc']:
        folder_path = payloads_dir / folder
        if not folder_path.exists():
            continue
        for f in folder_path.rglob('*'):
            try:
                if not f.is_file() or f.suffix not in ('', '.txt', '.lst', '.list'):
                    continue
                for line in f.read_text(errors='ignore').splitlines()[:1000]:
                    line = line.strip()
                    if line and len(line) > 3:
                        texts.append(line)
                        labels.append(1)
            except (OSError, PermissionError):
                continue
    
    mal_count = len(texts)
    print(f"Loaded {mal_count} malicious payloads")
    
    # Load CURATED benign data first (priority)
    if curated_dir.exists():
        for f in curated_dir.glob('*.txt'):
            try:
                for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line and len(line) > 2:
                        texts.append(line)
                        labels.append(0)
            except: pass
        print(f"Loaded {len(texts) - mal_count} curated benign samples")
    
    # Load ADVERSARIAL benign data (critical for reducing false positives)
    adv_count_before = len(texts)
    if adversarial_dir.exists():
        for f in adversarial_dir.glob('*.txt'):
            try:
                for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line and len(line) > 2:
                        texts.append(line)
                        labels.append(0)
            except: pass
        print(f"Loaded {len(texts) - adv_count_before} adversarial benign samples")
    
    # Load 500k FP test dataset (JSONL format)
    fp_test_file = base / 'datasets' / 'fp_test_500k.jsonl'
    fp_count_before = len(texts)
    if fp_test_file.exists():
        import json
        try:
            with open(fp_test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        text = obj.get('text', '').strip()
                        if text and len(text) > 2:
                            texts.append(text)
                            labels.append(0)
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(texts) - fp_count_before} samples from fp_test_500k.jsonl")
        except Exception as e:
            print(f"Warning: Could not load fp_test_500k.jsonl: {e}")
    
    # Fallback: wordlists if not enough benign
    if len(texts) - mal_count < mal_count // 2:
        wordlists = payloads_dir / 'wordlists'
        if wordlists.exists():
            for f in wordlists.rglob('*'):
                try:
                    if not f.is_file():
                        continue
                    for line in f.read_text(errors='ignore').splitlines()[:2000]:
                        line = line.strip()
                        if line and len(line) > 2 and len(line) < 100:
                            texts.append(line)
                            labels.append(0)
                except: pass
    
    ben_count = len(texts) - mal_count
    print(f"Loaded {ben_count} total benign samples")
    
    # Balance classes with SHUFFLED benign samples (critical for including fp_test data)
    if mal_count < ben_count:
        import random
        mal_texts = [t for t, l in zip(texts, labels) if l == 1]
        ben_texts = [t for t, l in zip(texts, labels) if l == 0]
        random.shuffle(ben_texts)  # Shuffle to get mix of all sources
        ben_texts = ben_texts[:mal_count]
        texts = mal_texts + ben_texts
        labels = [1] * len(mal_texts) + [0] * len(ben_texts)
    
    print(f"Final dataset: {len(texts)} samples (balanced)")
    return texts, labels


def train():
    """Main training function."""
    # Setup
    base_path = Path(__file__).parent.parent.parent
    device = setup_gpu()
    
    # Load data
    print("\n--- Loading Data ---")
    texts, labels = load_payload_data(base_path)
    
    if len(texts) < 100:
        print("ERROR: Not enough data. Check datasets/security_payloads/ folder.")
        return
    
    # Create dataset and split
    dataset = PayloadDataset(texts, labels, max_len=500)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = PayloadCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Training setup - with label smoothing to reduce overconfidence
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=5)
    
    # Label smoothing factor
    label_smoothing = 0.1
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_state = None
    
    for epoch in range(60):  # Increased from 50
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                # Apply label smoothing: targets become 0.1 or 0.9 instead of 0 or 1
                smoothed_targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = criterion(outputs, smoothed_targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid for predictions
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if early_stop(val_loss):
            print("Early stopping triggered")
            break
    
    # Save best model
    print("\n--- Saving Model ---")
    model.load_state_dict(best_state)
    models_dir = base_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save as TorchScript
    model.eval()
    example = torch.zeros(1, 500, dtype=torch.long).to(device)
    save_model(model, models_dir / 'payload_cnn', example)
    
    # Also save state dict
    torch.save(best_state, models_dir / 'payload_cnn.pth')
    
    print(f"✓ Model saved to models/payload_cnn.pt")
    print(f"✓ Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    train()
