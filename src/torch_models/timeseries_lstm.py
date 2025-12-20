"""LSTM model for time-series anomaly detection in network traffic."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesLSTM(nn.Module):
    """Bidirectional LSTM for temporal pattern detection."""
    
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # Logits


def get_model(input_dim=8, device='cuda'):
    """Create and return model on specified device."""
    return TimeSeriesLSTM(input_dim=input_dim).to(device)
