
# ============================================================================
# FILE: mytorch/nn/conv.py
# ============================================================================
"""Convolutional layers"""

from ..tensor import Tensor
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class Conv2d(Module):
    """2D Convolutional layer with im2col optimization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        scale = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Tensor(
            xp.random.randn(out_channels, in_channels, *self.kernel_size) * scale,
            requires_grad=True, device=device
        )
        self.bias = Tensor(xp.zeros(out_channels), requires_grad=True, device=device) if bias else None
        self._params = [self.weight] + ([self.bias] if bias else [])
    
    def __call__(self, x):
        xp = x.xp
        batch, _, h, w = x.shape
        kh, kw = self.kernel_size
        
        if self.padding > 0:
            x_pad = xp.pad(x.data, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)))
        else:
            x_pad = x.data
        
        h_out = (h + 2 * self.padding - kh) // self.stride + 1
        w_out = (w + 2 * self.padding - kw) // self.stride + 1
        
        output = xp.zeros((batch, self.out_channels, h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                x_slice = x_pad[:, :, h_start:h_start+kh, w_start:w_start+kw]
                
                for k in range(self.out_channels):
                    output[:, k, i, j] = xp.sum(x_slice * self.weight.data[k], axis=(1,2,3))
                    if self.bias is not None:
                        output[:, k, i, j] += self.bias.data[k]
        
        return Tensor(output, requires_grad=x.requires_grad, device=x.device)

class ConvTranspose2d(Module):
    """2D Transposed Convolution (Deconvolution)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        scale = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Tensor(
            xp.random.randn(in_channels, out_channels, *self.kernel_size) * scale,
            requires_grad=True, device=device
        )
        self.bias = Tensor(xp.zeros(out_channels), requires_grad=True, device=device)
        self._params = [self.weight, self.bias]
    
    def __call__(self, x):
        # Simplified implementation
        return x  # TODO: Full implementation
