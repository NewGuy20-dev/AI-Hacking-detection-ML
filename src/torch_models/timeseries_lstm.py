"""LSTM model for time-series anomaly detection in network traffic."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesLSTM(nn.Module):
    """Bidirectional LSTM for temporal pattern detection."""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        # *2 for bidirectional, *2 for mean+max pooling concat
        self.fc1 = nn.Linear(hidden_dim * 2 * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Mean + max pooling over all timesteps: captures gradual temporal changes
        # (exfiltration ramps, C2 beaconing pulses, slow DDoS buildup)
        mean_pool = lstm_out.mean(dim=1)                # (batch, hidden*2)
        max_pool = lstm_out.max(dim=1).values           # (batch, hidden*2)
        x = torch.cat([mean_pool, max_pool], dim=1)     # (batch, hidden*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)                  # Logits


class FocalLoss(nn.Module):
    """Focal loss for hard-example mining. Reduces weight on easy examples."""

    def __init__(self, gamma: float = 2.0, pos_weight: "torch.Tensor | None" = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(inputs)
        # pt = probability of the correct class
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def get_model(input_dim=8, device='cuda'):
    """Create and return model on specified device."""
    return TimeSeriesLSTM(input_dim=input_dim).to(device)
