
# ============================================================================
# FILE: mytorch/data/dataloader.py
# ============================================================================
"""DataLoader for batching and shuffling"""

import numpy as np
from ..tensor import Tensor

class DataLoader:
    """Data loader for batching, shuffling, and iterating over datasets"""
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
    
    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
            
            batch_indices = indices[start_idx:end_idx]
            batch = [self.dataset[i] for i in batch_indices]
            
            # Stack batch
            if isinstance(batch[0], tuple):
                yield tuple(self._stack([b[i] for b in batch]) for i in range(len(batch[0])))
            else:
                yield self._stack(batch)
    
    def _stack(self, items):
        if isinstance(items[0], Tensor):
            xp = items[0].xp
            device = items[0].device
            data = xp.stack([item.data for item in items])
            return Tensor(data, requires_grad=items[0].requires_grad, device=device)
        else:
            return np.stack(items)
    
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size