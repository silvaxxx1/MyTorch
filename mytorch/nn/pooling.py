
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
    """2D Max Pooling with backward"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        xp = x.xp
        batch, C, h, w = x.shape
        kh, kw = self.kernel_size
        s = self.stride

        h_out = (h - kh) // s + 1
        w_out = (w - kw) // s + 1

        output = xp.zeros((batch, C, h_out, w_out))
        # argmax row/col within input (absolute coordinates)
        max_h = xp.zeros((batch, C, h_out, w_out), dtype=int)
        max_w = xp.zeros((batch, C, h_out, w_out), dtype=int)

        b_idx = xp.arange(batch)[:, None]
        c_idx = xp.arange(C)[None, :]

        for i in range(h_out):
            for j in range(w_out):
                hs, ws = i * s, j * s
                patch = x.data[:, :, hs:hs + kh, ws:ws + kw]  # (batch, C, kh, kw)
                flat = patch.reshape(batch, C, -1).argmax(axis=2)  # (batch, C)
                dh, dw = flat // kw, flat % kw
                output[:, :, i, j] = patch[b_idx, c_idx, dh, dw]
                max_h[:, :, i, j] = hs + dh
                max_w[:, :, i, j] = ws + dw

        out = Tensor(output, requires_grad=x.requires_grad, device=x.device)
        if x.requires_grad:
            out._prev = {x}

            def _backward():
                dx = xp.zeros_like(x.data)
                # Build flat index arrays for scatter
                b_all = xp.arange(batch)[:, None, None, None] + xp.zeros((batch, C, h_out, w_out), dtype=int)
                c_all = xp.arange(C)[None, :, None, None] + xp.zeros((batch, C, h_out, w_out), dtype=int)
                xp.add.at(dx, (b_all, c_all, max_h, max_w), out.grad)
                x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out


class AvgPool2d(Module):
    """2D Average Pooling with backward"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        xp = x.xp
        batch, C, h, w = x.shape
        kh, kw = self.kernel_size
        s = self.stride

        h_out = (h - kh) // s + 1
        w_out = (w - kw) // s + 1

        output = xp.zeros((batch, C, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                hs, ws = i * s, j * s
                output[:, :, i, j] = x.data[:, :, hs:hs + kh, ws:ws + kw].mean(axis=(2, 3))

        out = Tensor(output, requires_grad=x.requires_grad, device=x.device)
        if x.requires_grad:
            out._prev = {x}
            pool_size = kh * kw

            def _backward():
                dx = xp.zeros_like(x.data)
                for i in range(h_out):
                    for j in range(w_out):
                        hs, ws = i * s, j * s
                        dx[:, :, hs:hs + kh, ws:ws + kw] += out.grad[:, :, i:i + 1, j:j + 1] / pool_size
                x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out


class AdaptiveAvgPool2d(Module):
    """Adaptive Average Pooling with backward"""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def __call__(self, x):
        xp = x.xp
        batch, C, h, w = x.shape
        h_out, w_out = self.output_size

        stride_h = h // h_out
        stride_w = w // w_out
        kernel_h = h - (h_out - 1) * stride_h
        kernel_w = w - (w_out - 1) * stride_w

        output = xp.zeros((batch, C, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride_h
                ws = j * stride_w
                output[:, :, i, j] = x.data[:, :, hs:hs + kernel_h, ws:ws + kernel_w].mean(axis=(2, 3))

        out = Tensor(output, requires_grad=x.requires_grad, device=x.device)
        if x.requires_grad:
            out._prev = {x}
            pool_size = kernel_h * kernel_w

            def _backward():
                dx = xp.zeros_like(x.data)
                for i in range(h_out):
                    for j in range(w_out):
                        hs = i * stride_h
                        ws = j * stride_w
                        dx[:, :, hs:hs + kernel_h, ws:ws + kernel_w] += (
                            out.grad[:, :, i:i + 1, j:j + 1] / pool_size
                        )
                x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out
