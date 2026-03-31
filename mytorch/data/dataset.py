
# ============================================================================
# FILE: mytorch/data/dataset.py
# ============================================================================
"""Dataset classes"""

class Dataset:
    """Abstract dataset class"""
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

class TensorDataset(Dataset):
    """Dataset wrapping tensors"""
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)
        self.tensors = tensors
    
    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)
    
    def __len__(self):
        return self.tensors[0].shape[0]