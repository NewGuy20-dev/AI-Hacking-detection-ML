"""Phase B1: Enhanced Payload CNN with attention mechanism and focal loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class PayloadCNNAttention(nn.Module):
    """Enhanced character-level CNN with self-attention for payload detection."""
    
    def __init__(self, vocab_size=256, embed_dim=128, num_filters=256, max_len=500, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv1d(embed_dim, num_filters // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters // 2, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=7, padding=3)
        
        # Residual connection
        self.residual = nn.Conv1d(embed_dim, num_filters, kernel_size=1)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(num_filters)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # x: (batch, seq_len) of char indices
        x = self.embedding(x)           # (batch, seq, embed)
        x = x.permute(0, 2, 1)          # (batch, embed, seq) for Conv1d
        
        # Residual path
        residual = self.residual(x)
        
        # Conv path
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Add residual
        x = x + residual
        
        # Self-attention (need batch, seq, features)
        x = x.permute(0, 2, 1)          # (batch, seq, filters)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        
        # Pool and classify
        x = x.permute(0, 2, 1)          # (batch, filters, seq)
        x = self.pool(x).squeeze(-1)    # (batch, num_filters)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


def get_model(device='cuda'):
    """Create and return enhanced model on specified device."""
    return PayloadCNNAttention().to(device)


def get_focal_loss(alpha=0.25, gamma=2.0):
    """Create focal loss criterion."""
    return FocalLoss(alpha, gamma)
