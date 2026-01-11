"""Train URL CNN model for malicious URL detection."""
import sys
import argparse
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from tqdm import tqdm

from torch_models.url_cnn import URLCNN
from torch_models.datasets import URLDataset
from torch_models.utils import setup_gpu, EarlyStopping, save_model
from training.checkpoint import CheckpointManager


def generate_malicious_urls(n=50000):
    """Generate synthetic malicious URLs with common attack patterns."""
    urls = []
    
    # Phishing patterns
    legit_brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'netflix', 'bank']
    tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.pw', '.cc', '.su']
    
    for _ in range(n // 5):
        brand = random.choice(legit_brands)
        tld = random.choice(tlds)
        # Typosquatting
        urls.append(f"http://{brand}{random.randint(1,99)}{tld}/login")
        # Subdomain abuse
        urls.append(f"http://{brand}.secure-login{tld}/verify")
        # Homograph-like
        urls.append(f"http://{brand.replace('a','4').replace('e','3')}.com/account")
        # Long subdomain
        urls.append(f"http://secure.{brand}.account.verify{tld}/auth")
        # IP-based
        urls.append(f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/{brand}/login")
    
    # Malware/exploit patterns
    for _ in range(n // 5):
        urls.append(f"http://download{random.randint(1,999)}.xyz/file.exe")
        urls.append(f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/malware.php")
        urls.append(f"http://free-software.tk/crack_{random.randint(1,100)}.zip")
        urls.append(f"http://update-flash.ml/player.exe")
        urls.append(f"http://cdn{random.randint(1,99)}.pw/script.js")
    
    # Encoded/obfuscated
    for _ in range(n // 10):
        urls.append(f"http://evil.com/%27%20OR%201=1--")
        urls.append(f"http://site.tk/page?id=<script>alert(1)</script>")
        urls.append(f"http://x.ml/r?u=http://phish.com")
        urls.append(f"http://bit.ly/{random.randint(100000,999999)}")
    
    return urls[:n]


def generate_benign_urls(n=50000):
    """Generate synthetic benign URLs."""
    urls = []
    
    domains = ['google.com', 'amazon.com', 'youtube.com', 'facebook.com', 'wikipedia.org',
               'twitter.com', 'instagram.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
               'reddit.com', 'netflix.com', 'microsoft.com', 'apple.com', 'yahoo.com']
    
    paths = ['', '/about', '/contact', '/products', '/services', '/blog', '/news', 
             '/help', '/support', '/login', '/signup', '/search', '/home', '/faq']
    
    for _ in range(n):
        domain = random.choice(domains)
        path = random.choice(paths)
        proto = random.choice(['http://', 'https://'])
        www = random.choice(['', 'www.'])
        
        if random.random() < 0.3:
            # Add query params
            params = f"?q={random.choice(['search', 'query', 'item'])}&page={random.randint(1,10)}"
        else:
            params = ''
        
        urls.append(f"{proto}{www}{domain}{path}{params}")
    
    return urls


def load_url_data(base_path):
    """Load URL data - uses real data + improved synthetic."""
    urls, labels = [], []
    
    url_dir = Path(base_path) / 'datasets' / 'url_analysis'
    
    # === REAL MALICIOUS URLs ===
    
    # Load URLhaus malicious URLs
    real_mal = url_dir / 'real_malicious_urls.txt'
    if real_mal.exists():
        for line in real_mal.read_text(errors='ignore').splitlines():
            if line.strip().startswith('http'):
                urls.append(line.strip())
                labels.append(1)
        print(f"Loaded {len(urls)} real malicious URLs (URLhaus)")
    
    # Load Kaggle malicious URLs (has both classes!)
    kaggle_file = url_dir / 'kaggle_malicious_urls.csv'
    if kaggle_file.exists():
        kaggle_mal, kaggle_ben = 0, 0
        for i, line in enumerate(kaggle_file.read_text(errors='ignore').splitlines()):
            if i == 0:  # Skip header
                continue
            parts = line.rsplit(',', 1)
            if len(parts) == 2:
                url, label = parts[0].strip(), parts[1].strip()
                if url and label in ('0', '1'):
                    urls.append(url if url.startswith('http') else f"http://{url}")
                    labels.append(int(label))
                    if label == '1':
                        kaggle_mal += 1
                    else:
                        kaggle_ben += 1
        print(f"Loaded {kaggle_mal} malicious + {kaggle_ben} benign from Kaggle dataset")
    
    # Load synthetic malicious (for diversity)
    synth_mal = url_dir / 'synthetic_malicious_hard.txt'
    if synth_mal.exists():
        mal_before = sum(labels)
        for line in synth_mal.read_text(encoding='utf-8', errors='ignore').splitlines()[:20000]:
            urls.append(line.strip())
            labels.append(1)
        print(f"Loaded {sum(labels) - mal_before} synthetic malicious URLs")
    
    mal_count = sum(labels)
    
    # === REAL BENIGN URLs ===
    
    # Load Common Crawl URLs from live_benign (NEW - up to 50M)
    live_benign_dir = Path(base_path) / 'datasets' / 'live_benign'
    cc_urls = live_benign_dir / 'common_crawl_urls.jsonl'
    if cc_urls.exists():
        ben_before = len(urls) - mal_count
        import json
        with open(cc_urls, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100000:  # Limit to 100K for balance
                    break
                try:
                    data = json.loads(line)
                    urls.append(data.get('text', ''))
                    labels.append(0)
                except:
                    pass
        print(f"Loaded {len(urls) - mal_count - ben_before} Common Crawl URLs (live_benign)")
    
    # Load Tranco top domains (REAL benign)
    tranco_file = url_dir / 'top-1m.csv'
    if tranco_file.exists():
        ben_before = len(urls) - mal_count
        for i, line in enumerate(tranco_file.read_text(errors='ignore').splitlines()):
            if i >= 50000:  # Limit to 50K
                break
            parts = line.split(',')
            if len(parts) >= 2:
                domain = parts[1].strip()
                if domain:
                    urls.append(f"https://{domain}/")
                    labels.append(0)
        print(f"Loaded {len(urls) - mal_count - ben_before} real benign domains (Tranco)")
    
    # Load synthetic benign (for diversity)
    synth_ben = url_dir / 'synthetic_benign_hard.txt'
    if synth_ben.exists():
        ben_before = len(urls) - mal_count
        for line in synth_ben.read_text(encoding='utf-8', errors='ignore').splitlines()[:20000]:
            urls.append(line.strip())
            labels.append(0)
        print(f"Loaded {len(urls) - mal_count - ben_before} synthetic benign URLs")
    
    # Fallback to generated if no data
    if len(urls) < 1000:
        print("Generating synthetic URLs (no real data found)...")
        mal_urls = generate_malicious_urls(50000)
        ben_urls = generate_benign_urls(50000)
        urls.extend(mal_urls)
        labels.extend([1] * len(mal_urls))
        urls.extend(ben_urls)
        labels.extend([0] * len(ben_urls))
    
    print(f"Total: {len(urls)} URLs ({sum(labels)} malicious, {len(labels)-sum(labels)} benign)")
    return urls, labels


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Save checkpoint every N batches')
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent
    device = setup_gpu()
    
    # Checkpoint manager
    ckpt_dir = base_path / 'checkpoints' / 'url'
    ckpt_mgr = CheckpointManager(str(ckpt_dir), 'url_cnn', args.checkpoint_every)
    
    # Load data
    print("\n--- Loading Data ---")
    urls, labels = load_url_data(base_path)
    
    # Create dataset and split
    dataset = URLDataset(urls, labels, max_len=200)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, timeout=0, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, timeout=0, persistent_workers=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = URLCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=5)
    
    # Resume from checkpoint if requested
    start_epoch, start_batch, global_step = 0, 0, 0
    if args.resume or ckpt_mgr.find_latest():
        resume_info = ckpt_mgr.load(model, optimizer, scheduler, scaler, device)
        start_epoch = resume_info['epoch']
        start_batch = resume_info['batch_idx']
        global_step = resume_info['global_step']
        if start_batch >= len(train_loader):
            start_epoch += 1
            start_batch = 0
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_state = None
    
    for epoch in range(start_epoch, 60):
        # Train
        model.train()
        train_loss = 0
        batches_processed = 0
        
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    initial=epoch_start_batch, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in pbar:
            if batch_idx < epoch_start_batch:
                continue
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            batches_processed += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if ckpt_mgr.should_save(batch_idx):
                ckpt_mgr.save(epoch, batch_idx, model, optimizer, scheduler, scaler, global_step)
        
        ckpt_mgr.save(epoch, len(train_loader), model, optimizer, scheduler, scaler, global_step)
        
        train_loss /= max(batches_processed, 1)
        
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
                preds = (torch.sigmoid(outputs) > 0.5).float()
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
    
    model.eval()
    example = torch.zeros(1, 200, dtype=torch.long).to(device)
    save_model(model, models_dir / 'url_cnn', example)
    torch.save(best_state, models_dir / 'url_cnn.pth')
    
    print(f"✓ Model saved to models/url_cnn.pt")
    print(f"✓ Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    train()
