

# ============================================================================
# FILE: mytorch/nn/dropout.py
# ============================================================================
"""Dropout layers"""

from ..tensor import Tensor
from .modules import Module

class Dropout(Module):
    """Dropout regularization"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        xp = x.xp
        mask = xp.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        return Tensor(x.data * mask, requires_grad=x.requires_grad, device=x.device)

class Dropout2d(Dropout):
    """2D Dropout (drops entire channels)"""
    pass
