"""CNN model for payload/attack detection (SQL injection, XSS, command injection)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PayloadCNN(nn.Module):
    """Character-level CNN for malicious payload detection."""
    
    def __init__(self, vocab_size=256, embed_dim=128, num_filters=256, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(embed_dim, num_filters // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters // 2, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # x: (batch, seq_len) of char indices
        x = self.embedding(x)           # (batch, seq, embed)
        x = x.permute(0, 2, 1)          # (batch, embed, seq) for Conv1d
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)    # (batch, num_filters)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # Return logits, no sigmoid


def get_model(device='cuda'):
    """Create and return model on specified device."""
    model = PayloadCNN()
    return model.to(device)
