
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


class BatchNorm1d(Module):
    """1D Batch Normalization with backward"""

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

        std = xp.sqrt(var + self.eps)
        x_hat = (x.data - mean) / std
        out_data = self.gamma.data * x_hat + self.beta.data

        req = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        out = Tensor(out_data, requires_grad=req, device=x.device)

        if req:
            out._prev = {x, self.gamma, self.beta}
            N = x.data.shape[0]

            def _backward():
                dy = out.grad  # (N, features)

                if self.gamma.requires_grad:
                    dgamma = (dy * x_hat).sum(axis=0)
                    self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma

                if self.beta.requires_grad:
                    dbeta = dy.sum(axis=0)
                    self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta

                if x.requires_grad:
                    dx_hat = dy * self.gamma.data
                    dvar = (dx_hat * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5).sum(axis=0)
                    dmu = (-dx_hat / std).sum(axis=0) + dvar * (-2 * (x.data - mean)).sum(axis=0) / N
                    dx = dx_hat / std + dvar * 2 * (x.data - mean) / N + dmu / N
                    x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out


class BatchNorm2d(Module):
    """2D Batch Normalization with backward (normalizes over N, H, W per channel)"""

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
        # x: (N, C, H, W)
        if self.training:
            mean = x.data.mean(axis=(0, 2, 3))        # (C,)
            var = x.data.var(axis=(0, 2, 3))           # (C,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std = xp.sqrt(var + self.eps)
        x_hat = (x.data - mean[None, :, None, None]) / std[None, :, None, None]
        out_data = self.gamma.data[None, :, None, None] * x_hat + self.beta.data[None, :, None, None]

        req = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        out = Tensor(out_data, requires_grad=req, device=x.device)

        if req:
            out._prev = {x, self.gamma, self.beta}
            N_total = x.data.shape[0] * x.data.shape[2] * x.data.shape[3]  # N*H*W

            def _backward():
                dy = out.grad  # (N, C, H, W)

                if self.gamma.requires_grad:
                    dgamma = (dy * x_hat).sum(axis=(0, 2, 3))
                    self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma

                if self.beta.requires_grad:
                    dbeta = dy.sum(axis=(0, 2, 3))
                    self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta

                if x.requires_grad:
                    g = self.gamma.data[None, :, None, None]
                    s = std[None, :, None, None]
                    m = mean[None, :, None, None]
                    v = var[None, :, None, None]
                    dx_hat = dy * g
                    dvar = (dx_hat * (x.data - m) * -0.5 * (v + self.eps) ** -1.5).sum(axis=(0, 2, 3), keepdims=True)
                    dmu = (-dx_hat / s).sum(axis=(0, 2, 3), keepdims=True) + dvar * (-2 * (x.data - m)).sum(axis=(0, 2, 3), keepdims=True) / N_total
                    dx = dx_hat / s + dvar * 2 * (x.data - m) / N_total + dmu / N_total
                    x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out


class LayerNorm(Module):
    """Layer Normalization with backward"""

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
        std = xp.sqrt(var + self.eps)
        x_hat = (x.data - mean) / std
        out_data = self.gamma.data * x_hat + self.beta.data

        req = x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        out = Tensor(out_data, requires_grad=req, device=x.device)

        if req:
            out._prev = {x, self.gamma, self.beta}
            N = x.data.shape[-1]

            def _backward():
                dy = out.grad
                reduce_axes = tuple(range(dy.ndim - 1))

                if self.gamma.requires_grad:
                    dgamma = (dy * x_hat).sum(axis=reduce_axes)
                    self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma

                if self.beta.requires_grad:
                    dbeta = dy.sum(axis=reduce_axes)
                    self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta

                if x.requires_grad:
                    dx_hat = dy * self.gamma.data
                    dvar = (dx_hat * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5).sum(axis=-1, keepdims=True)
                    dmu = (-dx_hat / std).sum(axis=-1, keepdims=True) + dvar * (-2 * (x.data - mean)).sum(axis=-1, keepdims=True) / N
                    dx = dx_hat / std + dvar * 2 * (x.data - mean) / N + dmu / N
                    x.grad = dx if x.grad is None else x.grad + dx

            out._grad_fn = _backward
        return out
