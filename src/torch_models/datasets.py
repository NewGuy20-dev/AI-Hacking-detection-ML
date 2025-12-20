"""PyTorch Dataset classes for security data."""
import torch
from torch.utils.data import Dataset
import numpy as np


class PayloadDataset(Dataset):
    """Dataset for payload/content classification."""
    
    def __init__(self, texts, labels, max_len=500):
        self.texts = texts
        self.labels = np.array(labels, dtype=np.float32)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Convert to character indices (0-255)
        chars = [ord(c) % 256 for c in text[:self.max_len]]
        # Pad to max_len
        chars += [0] * (self.max_len - len(chars))
        return {
            'input': torch.tensor(chars, dtype=torch.long),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class URLDataset(Dataset):
    """Dataset for URL classification."""
    
    def __init__(self, urls, labels, max_len=200):
        self.urls = urls
        self.labels = np.array(labels, dtype=np.float32)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = str(self.urls[idx])
        # Convert to character indices (0-127 ASCII)
        chars = [ord(c) % 128 for c in url[:self.max_len]]
        chars += [0] * (self.max_len - len(chars))
        return {
            'input': torch.tensor(chars, dtype=torch.long),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class TimeSeriesDataset(Dataset):
    """Dataset for time-series anomaly detection."""
    
    def __init__(self, sequences, labels):
        self.sequences = np.array(sequences, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
