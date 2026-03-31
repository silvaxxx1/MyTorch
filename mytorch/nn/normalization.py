
# ============================================================================
# FILE: mytorch/nn/normalization.py  
# ============================================================================
"""Normalization layers"""

from ..tensor import Tensor
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class LayerNorm(Module):
    """Layer Normalization (critical for transformers)"""
    def __init__(self, normalized_shape, eps=1e-5, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(xp.ones(normalized_shape), requires_grad=True, device=device)
        self.beta = Tensor(xp.zeros(normalized_shape), requires_grad=True, device=device)
        self._params = [self.gamma, self.beta]
    
    def __call__(self, x):
        xp = x.xp
        mean = xp.mean(x.data, axis=-1, keepdims=True)
        var = xp.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / xp.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data
        return Tensor(out, requires_grad=x.requires_grad, device=x.device)

class BatchNorm1d(Module):
    """1D Batch Normalization"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(xp.ones(num_features), requires_grad=True, device=device)
        self.beta = Tensor(xp.zeros(num_features), requires_grad=True, device=device)
        self.running_mean = xp.zeros(num_features)
        self.running_var = xp.ones(num_features)
        self._params = [self.gamma, self.beta]
    
    def __call__(self, x):
        xp = x.xp
        if self.training:
            mean = xp.mean(x.data, axis=0)
            var = xp.var(x.data, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x.data - mean) / xp.sqrt(var + self.eps)
        out = self.gamma.data * x_norm + self.beta.data
        return Tensor(out, requires_grad=x.requires_grad, device=x.device)

class BatchNorm2d(BatchNorm1d):
    """2D Batch Normalization"""
    pass

class GroupNorm(Module):
    """Group Normalization"""
    def __init__(self, num_groups, num_channels, eps=1e-5, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = Tensor(xp.ones(num_channels), requires_grad=True, device=device)
        self.beta = Tensor(xp.zeros(num_channels), requires_grad=True, device=device)
        self._params = [self.gamma, self.beta]
    
    def __call__(self, x):
        # Simplified implementation
        return x