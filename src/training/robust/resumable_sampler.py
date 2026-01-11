"""Resumable sampler for mid-epoch checkpoint recovery."""
from typing import Iterator, List, Optional, Dict, Any
import torch
from torch.utils.data import Sampler


class ResumableSampler(Sampler[int]):
    """Sampler that supports resuming from a specific position.
    
    Features:
    - Deterministic shuffling with seed for reproducibility
    - Resume from arbitrary position (for mid-epoch checkpoints)
    - State serialization for checkpointing
    """
    
    def __init__(
        self,
        data_source,
        shuffle: bool = True,
        seed: int = 42,
        start_index: int = 0,
        indices: Optional[List[int]] = None,
    ):
        """
        Args:
            data_source: Dataset to sample from
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling (for reproducibility)
            start_index: Index to start iteration from (for resume)
            indices: Pre-computed indices (for restoring from checkpoint)
        """
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.start_index = start_index
        
        # Use provided indices or generate new ones
        if indices is not None:
            self._indices = list(indices)
        else:
            self._indices = self._generate_indices()
    
    def _generate_indices(self) -> List[int]:
        """Generate index permutation."""
        n = len(self.data_source)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            return torch.randperm(n, generator=g).tolist()
        return list(range(n))
    
    def __iter__(self) -> Iterator[int]:
        """Iterate from start_index to end."""
        return iter(self._indices[self.start_index:])
    
    def __len__(self) -> int:
        """Return remaining samples from start_index."""
        return len(self._indices) - self.start_index
    
    @property
    def total_length(self) -> int:
        """Return total samples (ignoring start_index)."""
        return len(self._indices)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling (changes seed)."""
        self.seed = self.seed + epoch
        self._indices = self._generate_indices()
        self.start_index = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'shuffle': self.shuffle,
            'seed': self.seed,
            'start_index': self.start_index,
            'indices': self._indices,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.shuffle = state['shuffle']
        self.seed = state['seed']
        self.start_index = state['start_index']
        self._indices = state['indices']
    
    @classmethod
    def from_state_dict(cls, data_source, state: Dict[str, Any]) -> 'ResumableSampler':
        """Create sampler from checkpoint state."""
        return cls(
            data_source=data_source,
            shuffle=state['shuffle'],
            seed=state['seed'],
            start_index=state['start_index'],
            indices=state['indices'],
        )


class ResumableDistributedSampler(Sampler[int]):
    """Distributed sampler with resume support for multi-GPU training."""
    
    def __init__(
        self,
        data_source,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 42,
        start_index: int = 0,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.start_index = start_index
        self.drop_last = drop_last
        
        # Calculate samples per replica
        total = len(data_source)
        if drop_last and total % num_replicas != 0:
            self.num_samples = total // num_replicas
        else:
            self.num_samples = (total + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        
        self._indices = self._generate_indices()
    
    def _generate_indices(self) -> List[int]:
        """Generate indices for this rank."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        if self.shuffle:
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            indices = list(range(len(self.data_source)))
        
        # Pad to make evenly divisible
        padding = self.total_size - len(indices)
        if padding > 0:
            indices += indices[:padding]
        
        # Subsample for this rank
        return indices[self.rank:self.total_size:self.num_replicas]
    
    def __iter__(self) -> Iterator[int]:
        return iter(self._indices[self.start_index:])
    
    def __len__(self) -> int:
        return self.num_samples - self.start_index
    
    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch
        self._indices = self._generate_indices()
        self.start_index = 0
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'num_replicas': self.num_replicas,
            'rank': self.rank,
            'shuffle': self.shuffle,
            'seed': self.seed,
            'start_index': self.start_index,
            'drop_last': self.drop_last,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.seed = state['seed']
        self.start_index = state['start_index']
        self._indices = self._generate_indices()
