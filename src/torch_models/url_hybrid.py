"""Phase B2: Hybrid URL model combining CNN with engineered features."""
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_url_features(url):
    """Extract engineered features from URL."""
    url = str(url)
    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
    
    # Basic features
    features = {
        'length': min(len(url) / 200, 1.0),
        'num_dots': min(url.count('.') / 10, 1.0),
        'num_hyphens': min(url.count('-') / 5, 1.0),
        'num_underscores': min(url.count('_') / 5, 1.0),
        'num_digits': min(sum(c.isdigit() for c in url) / 20, 1.0),
        'num_params': min(url.count('&') / 5, 1.0),
        'has_ip': float(bool(re.match(r'https?://\d+\.\d+\.\d+\.\d+', url))),
        'has_https': float(url.startswith('https')),
        'path_depth': min(url.count('/') / 10, 1.0),
        'has_at': float('@' in url),
        'has_double_slash': float('//' in url[8:]),  # After protocol
        'subdomain_depth': min(domain.count('.') / 5, 1.0),
    }
    
    # Suspicious TLD check
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.pw', '.cc', '.su']
    features['suspicious_tld'] = float(any(url.lower().endswith(tld) for tld in suspicious_tlds))
    
    # Entropy of domain (high entropy = random/suspicious)
    if domain:
        prob = [domain.count(c) / len(domain) for c in set(domain)]
        entropy = -sum(p * math.log2(p) for p in prob if p > 0)
        features['domain_entropy'] = min(entropy / 5, 1.0)
    else:
        features['domain_entropy'] = 0.0
    
    # Suspicious keywords
    suspicious_words = ['login', 'verify', 'secure', 'account', 'update', 'confirm', 'bank', 'paypal']
    features['suspicious_keywords'] = min(sum(w in url.lower() for w in suspicious_words) / 3, 1.0)
    
    return list(features.values())


class HybridURLModel(nn.Module):
    """Hybrid model combining CNN character features with engineered features."""
    
    def __init__(self, vocab_size=128, embed_dim=64, max_len=200, num_features=15):
        super().__init__()
        
        # CNN branch (character-level)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.cnn_fc = nn.Linear(64 * 3, 64)
        
        # Feature branch (engineered features)
        self.feature_fc = nn.Linear(num_features, 32)
        
        # Combined classifier
        self.combined_fc1 = nn.Linear(64 + 32, 48)
        self.dropout = nn.Dropout(0.3)
        self.combined_fc2 = nn.Linear(48, 1)
    
    def forward(self, char_input, features):
        # CNN branch
        x = self.embedding(char_input).permute(0, 2, 1)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [self.pool(c).squeeze(-1) for c in conv_outs]
        cnn_out = F.relu(self.cnn_fc(torch.cat(pooled, dim=1)))
        
        # Feature branch
        feat_out = F.relu(self.feature_fc(features))
        
        # Combine
        combined = torch.cat([cnn_out, feat_out], dim=1)
        x = F.relu(self.combined_fc1(combined))
        x = self.dropout(x)
        return self.combined_fc2(x).squeeze(-1)


def get_model(device='cuda'):
    """Create and return hybrid model on specified device."""
    return HybridURLModel().to(device)


def tokenize_url(url, max_len=200):
    """Convert URL to character indices."""
    chars = [ord(c) % 128 for c in str(url)[:max_len]]
    chars += [0] * (max_len - len(chars))
    return chars
