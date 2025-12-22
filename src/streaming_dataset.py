"""Memory-efficient streaming dataset for large text files."""
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """Memory-efficient dataset that streams lines from text files."""
    
    def __init__(self, file_paths: Union[str, Path, List], 
                 transform: Optional[Callable] = None,
                 skip_empty: bool = True,
                 encoding: str = 'utf-8'):
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        self.file_paths = [Path(p) for p in file_paths]
        self.transform = transform
        self.skip_empty = skip_empty
        self.encoding = encoding
    
    def __iter__(self) -> Iterator:
        for path in self.file_paths:
            if not path.exists():
                continue
            try:
                with open(path, 'r', encoding=self.encoding, errors='ignore') as f:
                    for line in f:
                        line = line.rstrip('\n\r')
                        if self.skip_empty and not line:
                            continue
                        yield self.transform(line) if self.transform else line
            except (OSError, PermissionError):
                continue


class LabeledStreamingDataset(IterableDataset):
    """Streaming dataset with labels from (file_path, label) pairs."""
    
    def __init__(self, file_label_pairs: List[Tuple[Union[str, Path], int]],
                 transform: Optional[Callable] = None,
                 skip_empty: bool = True,
                 encoding: str = 'utf-8'):
        self.file_label_pairs = [(Path(p), label) for p, label in file_label_pairs]
        self.transform = transform
        self.skip_empty = skip_empty
        self.encoding = encoding
    
    def __iter__(self) -> Iterator[Tuple]:
        for path, label in self.file_label_pairs:
            if not path.exists():
                continue
            try:
                with open(path, 'r', encoding=self.encoding, errors='ignore') as f:
                    for line in f:
                        line = line.rstrip('\n\r')
                        if self.skip_empty and not line:
                            continue
                        text = self.transform(line) if self.transform else line
                        yield text, label
            except (OSError, PermissionError):
                continue


def create_dataloader(file_paths: List, transform: Callable = None, 
                      batch_size: int = 32, labeled: bool = False,
                      labels: List[int] = None) -> torch.utils.data.DataLoader:
    """Create a DataLoader from file paths."""
    if labeled and labels:
        dataset = LabeledStreamingDataset(list(zip(file_paths, labels)), transform=transform)
    else:
        dataset = StreamingDataset(file_paths, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
