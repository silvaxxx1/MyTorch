# ============================================================================
# FILE: mytorch/nn/embedding.py
# ============================================================================
"""Embedding layer"""

from ..tensor import Tensor
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class Embedding(Module):
    """Embedding layer for discrete tokens"""
    def __init__(self, num_embeddings, embedding_dim, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        scale = xp.sqrt(1.0 / embedding_dim)
        self.weight = Tensor(xp.random.randn(num_embeddings, embedding_dim) * scale,
                            requires_grad=True, device=device)
        self._params = [self.weight]
    
    def __call__(self, x):
        # x: indices of shape (batch, seq_len)
        xp = self.weight.xp
        indices = x.data if isinstance(x, Tensor) else x
        embeddings = self.weight.data[indices.astype(int)]
        return Tensor(embeddings, requires_grad=True, device=self.weight.device)
