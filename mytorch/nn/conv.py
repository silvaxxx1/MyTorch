
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
    """2D Convolutional layer (im2col + matmul, with full backward)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.device = device

        scale = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Tensor(
            xp.random.randn(out_channels, in_channels, *self.kernel_size) * scale,
            requires_grad=True, device=device
        )
        self.bias = Tensor(xp.zeros(out_channels), requires_grad=True, device=device) if bias else None
        self._params = [self.weight] + ([self.bias] if bias else [])

    def __call__(self, x):
        xp = x.xp
        batch, C_in, h, w = x.shape
        kh, kw = self.kernel_size
        s, p = self.stride, self.padding

        if p > 0:
            x_pad = xp.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            x_pad = x.data

        h_out = (h + 2 * p - kh) // s + 1
        w_out = (w + 2 * p - kw) // s + 1

        # im2col: (batch, C_in*kh*kw, h_out*w_out)
        col = xp.zeros((batch, C_in * kh * kw, h_out * w_out))
        for i in range(h_out):
            for j in range(w_out):
                patch = x_pad[:, :, i * s:i * s + kh, j * s:j * s + kw]
                col[:, :, i * w_out + j] = patch.reshape(batch, -1)

        # W: (C_out, C_in*kh*kw)
        W = self.weight.data.reshape(self.out_channels, -1)
        # output: (batch, C_out, h_out, w_out)
        out_data = (W @ col).reshape(batch, self.out_channels, h_out, w_out)
        if self.bias is not None:
            out_data = out_data + self.bias.data[None, :, None, None]

        req = x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad)
        out = Tensor(out_data, requires_grad=req, device=x.device)

        if req:
            out._prev = {x, self.weight} | ({self.bias} if self.bias is not None else set())

            def _backward():
                grad = out.grad  # (batch, C_out, h_out, w_out)
                grad_2d = grad.reshape(batch, self.out_channels, -1)  # (batch, C_out, h_out*w_out)

                if self.weight.requires_grad:
                    dW = xp.sum(grad_2d @ col.transpose(0, 2, 1), axis=0)  # (C_out, C_in*kh*kw)
                    self.weight.grad = (dW.reshape(self.weight.shape) if self.weight.grad is None
                                        else self.weight.grad + dW.reshape(self.weight.shape))

                if self.bias is not None and self.bias.requires_grad:
                    db = grad.sum(axis=(0, 2, 3))
                    self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

                if x.requires_grad:
                    # dcol: (batch, C_in*kh*kw, h_out*w_out)
                    dcol = (grad_2d.transpose(0, 2, 1) @ W).transpose(0, 2, 1)

                    # col2im
                    dx_pad = xp.zeros_like(x_pad)
                    for i in range(h_out):
                        for j in range(w_out):
                            patch_grad = dcol[:, :, i * w_out + j].reshape(batch, C_in, kh, kw)
                            dx_pad[:, :, i * s:i * s + kh, j * s:j * s + kw] += patch_grad

                    dx = dx_pad[:, :, p:-p, p:-p] if p > 0 else dx_pad
                    x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out
