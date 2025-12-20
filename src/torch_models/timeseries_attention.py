"""Phase B3: AttentionLSTM for time-series anomaly detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with temporal attention for anomaly detection."""
    
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum (context vector)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)
    
    def get_attention_weights(self, x):
        """Return attention weights for visualization."""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_weights = self.attention(lstm_out)
            return F.softmax(attn_weights, dim=1).squeeze(-1)


class TimeSeriesAutoencoder(nn.Module):
    """Autoencoder for reconstruction-based anomaly detection."""
    
    def __init__(self, input_dim=8, hidden_dim=32, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Encode
        _, (h, c) = self.encoder(x)
        
        # Decode (use last hidden state repeated)
        decoder_input = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input)
        
        return self.output(decoder_out)
    
    def anomaly_score(self, x):
        """Compute reconstruction error as anomaly score."""
        recon = self.forward(x)
        mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse


def get_model(input_dim=8, device='cuda'):
    """Create and return AttentionLSTM on specified device."""
    return AttentionLSTM(input_dim=input_dim).to(device)


def get_autoencoder(input_dim=8, seq_len=60, device='cuda'):
    """Create and return autoencoder on specified device."""
    return TimeSeriesAutoencoder(input_dim=input_dim, seq_len=seq_len).to(device)
