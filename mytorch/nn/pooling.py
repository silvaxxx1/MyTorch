
# ============================================================================
# FILE: mytorch/nn/pooling.py
# ============================================================================
"""Pooling layers"""

from ..tensor import Tensor
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class MaxPool2d(Module):
    """2D Max Pooling"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or kernel_size
        self.padding = padding
    
    def __call__(self, x):
        xp = x.xp
        batch, channels, h, w = x.shape
        kh, kw = self.kernel_size
        
        h_out = (h - kh) // self.stride + 1
        w_out = (w - kw) // self.stride + 1
        
        output = xp.zeros((batch, channels, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                x_slice = x.data[:, :, h_start:h_start+kh, w_start:w_start+kw]
                output[:, :, i, j] = xp.max(x_slice, axis=(2,3))
        
        return Tensor(output, requires_grad=x.requires_grad, device=x.device)

class AvgPool2d(Module):
    """2D Average Pooling"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or kernel_size
        self.padding = padding
    
    def __call__(self, x):
        xp = x.xp
        batch, channels, h, w = x.shape
        kh, kw = self.kernel_size
        
        h_out = (h - kh) // self.stride + 1
        w_out = (w - kw) // self.stride + 1
        
        output = xp.zeros((batch, channels, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                x_slice = x.data[:, :, h_start:h_start+kh, w_start:w_start+kw]
                output[:, :, i, j] = xp.mean(x_slice, axis=(2,3))
        
        return Tensor(output, requires_grad=x.requires_grad, device=x.device)

class AdaptiveAvgPool2d(Module):
    """Adaptive Average Pooling"""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    def __call__(self, x):
        xp = x.xp
        batch, channels, h, w = x.shape
        h_out, w_out = self.output_size
        
        stride_h = h // h_out
        stride_w = w // w_out
        kernel_h = h - (h_out - 1) * stride_h
        kernel_w = w - (w_out - 1) * stride_w
        
        output = xp.zeros((batch, channels, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride_h
                w_start = j * stride_w
                h_end = h_start + kernel_h
                w_end = w_start + kernel_w
                output[:, :, i, j] = xp.mean(x.data[:, :, h_start:h_end, w_start:w_end], axis=(2,3))
        
        return Tensor(output, requires_grad=x.requires_grad, device=x.device)
