
# ============================================================================
# FILE: mytorch/functional.py
# ============================================================================
"""Functional API for activations and loss functions"""

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from .tensor import Tensor

# Activation Functions
def relu(x):
    xp = x.xp
    out = Tensor(xp.maximum(0, x.data), requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * (x.data > 0)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def leaky_relu(x, alpha=0.01):
    xp = x.xp
    out = Tensor(xp.where(x.data > 0, x.data, alpha * x.data), requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * xp.where(x.data > 0, 1.0, alpha)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def elu(x, alpha=1.0):
    xp = x.xp
    out = Tensor(xp.where(x.data > 0, x.data, alpha * (xp.exp(x.data) - 1)), requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * xp.where(x.data > 0, 1.0, alpha * xp.exp(x.data))
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def gelu(x):
    """GELU activation (used in transformers)"""
    xp = x.xp
    sqrt_2_pi = xp.sqrt(2.0 / xp.pi)
    cdf = 0.5 * (1.0 + xp.tanh(sqrt_2_pi * (x.data + 0.044715 * x.data ** 3)))
    out = Tensor(x.data * cdf, requires_grad=x.requires_grad, device=x.device)
    
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                tanh_arg = sqrt_2_pi * (x.data + 0.044715 * x.data ** 3)
                tanh_val = xp.tanh(tanh_arg)
                sech2 = 1 - tanh_val ** 2
                grad_cdf = 0.5 * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * x.data ** 2)
                grad = out.grad * (cdf + x.data * grad_cdf)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def sigmoid(x):
    xp = x.xp
    sig = 1 / (1 + xp.exp(-x.data))
    out = Tensor(sig, requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * sig * (1 - sig)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def tanh(x):
    xp = x.xp
    tanh_val = xp.tanh(x.data)
    out = Tensor(tanh_val, requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * (1 - tanh_val ** 2)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def softmax(x, axis=-1):
    xp = x.xp
    exp_x = xp.exp(x.data - xp.max(x.data, axis=axis, keepdims=True))
    sm = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(sm, requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                grad = out.grad * sm
                sum_grad = xp.sum(grad, axis=axis, keepdims=True)
                grad = grad - sm * sum_grad
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

def log_softmax(x, axis=-1):
    xp = x.xp
    max_x = xp.max(x.data, axis=axis, keepdims=True)
    lse = max_x + xp.log(xp.sum(xp.exp(x.data - max_x), axis=axis, keepdims=True))
    out = Tensor(x.data - lse, requires_grad=x.requires_grad, device=x.device)
    if out.requires_grad:
        out._prev = {x}
        def _backward():
            if x.requires_grad:
                sm = xp.exp(out.data)
                grad = out.grad - sm * xp.sum(out.grad, axis=axis, keepdims=True)
                x.grad = grad if x.grad is None else x.grad + grad
        out._grad_fn = _backward
    return out

# Loss Functions
def mse_loss(pred, target):
    """Mean Squared Error"""
    diff = pred - target
    return (diff * diff).mean()

def mae_loss(pred, target):
    """Mean Absolute Error"""
    xp = pred.xp
    diff = pred.data - target.data
    abs_diff = xp.abs(diff)
    out = Tensor(abs_diff.mean(), requires_grad=pred.requires_grad or target.requires_grad, device=pred.device)
    if out.requires_grad:
        out._prev = {pred, target}
        n = diff.size
        def _backward():
            grad = out.grad * xp.sign(diff) / n
            if pred.requires_grad:
                pred.grad = grad if pred.grad is None else pred.grad + grad
            if target.requires_grad:
                target.grad = -grad if target.grad is None else target.grad - grad
        out._grad_fn = _backward
    return out

def cross_entropy(pred, target):
    """Cross Entropy Loss"""
    log_probs = log_softmax(pred, axis=-1)
    xp = pred.xp
    batch_size = pred.shape[0]
    target_data = target.data if isinstance(target, Tensor) else target
    indices = target_data.astype(int)
    selected = log_probs[xp.arange(batch_size), indices]
    return -selected.mean()

def binary_cross_entropy(pred, target):
    """Binary Cross Entropy"""
    xp = pred.xp
    eps = 1e-7
    pred_data = xp.clip(pred.data, eps, 1 - eps)
    loss_data = -(target.data * xp.log(pred_data) + (1 - target.data) * xp.log(1 - pred_data))
    out = Tensor(loss_data.mean(), requires_grad=pred.requires_grad, device=pred.device)
    if out.requires_grad:
        out._prev = {pred}
        n = pred.data.size
        def _backward():
            grad = out.grad * (-target.data / pred_data + (1 - target.data) / (1 - pred_data)) / n
            pred.grad = grad if pred.grad is None else pred.grad + grad
        out._grad_fn = _backward
    return out

def huber_loss(pred, target, delta=1.0):
    """Huber Loss (smooth L1)"""
    xp = pred.xp
    diff = pred.data - target.data
    abs_diff = xp.abs(diff)
    loss_data = xp.where(abs_diff < delta, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    out = Tensor(loss_data.mean(), requires_grad=pred.requires_grad, device=pred.device)
    if out.requires_grad:
        out._prev = {pred}
        n = pred.data.size
        def _backward():
            grad = out.grad * xp.where(abs_diff < delta, diff, delta * xp.sign(diff)) / n
            pred.grad = grad if pred.grad is None else pred.grad + grad
        out._grad_fn = _backward
    return out