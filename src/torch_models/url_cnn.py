"""CNN model for malicious URL detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class URLCNN(nn.Module):
    """Character-level CNN for URL classification."""
    
    def __init__(self, vocab_size=128, embed_dim=64, max_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-kernel convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64 * 3, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed, seq)
        
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [self.pool(c).squeeze(-1) for c in conv_outs]
        
        x = torch.cat(pooled, dim=1)  # (batch, 64*3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # Logits


def get_model(device='cuda'):
    """Create and return model on specified device."""
    return URLCNN().to(device)
