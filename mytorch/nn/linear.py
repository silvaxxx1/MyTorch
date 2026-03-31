
# ============================================================================
# FILE: mytorch/nn/linear.py
# ============================================================================
"""Linear layers"""

from ..tensor import Tensor
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class Linear(Module):
    """Fully connected layer"""
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        scale = xp.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(xp.random.randn(in_features, out_features) * scale, 
                            requires_grad=True, device=device)
        self.bias = Tensor(xp.zeros(out_features), requires_grad=True, device=device) if bias else None
        self._params = [self.weight] + ([self.bias] if bias else [])
    
    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out