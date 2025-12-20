"""Neural meta-classifier that combines outputs from all detection models."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaClassifier(nn.Module):
    """Learns optimal combination of model outputs for final prediction."""
    
    def __init__(self, num_models=5):
        super().__init__()
        self.fc1 = nn.Linear(num_models, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        # x: (batch, num_models) - probability scores from each model
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # Logits


def get_model(num_models=5, device='cuda'):
    """Create and return model on specified device."""
    return MetaClassifier(num_models=num_models).to(device)
