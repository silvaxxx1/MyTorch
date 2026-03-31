
# ============================================================================
# FILE: mytorch/nn/activation.py
# ============================================================================
"""Activation function modules"""

from ..tensor import Tensor
from .. import functional as F
from .modules import Module

class ReLU(Module):
    def __call__(self, x):
        return F.relu(x)

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    
    def __call__(self, x):
        return F.leaky_relu(x, self.alpha)

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def __call__(self, x):
        return F.elu(x, self.alpha)

class GELU(Module):
    def __call__(self, x):
        return F.gelu(x)

class Sigmoid(Module):
    def __call__(self, x):
        return F.sigmoid(x)

class Tanh(Module):
    def __call__(self, x):
        return F.tanh(x)

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
    
    def __call__(self, x):
        return F.softmax(x, self.axis)