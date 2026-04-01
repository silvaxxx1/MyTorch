
# ============================================================================
# FILE: mytorch/tensor.py
# ============================================================================
"""Tensor implementation with autograd"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class Tensor:
    """Tensor with automatic differentiation"""
    
    def __init__(self, data, requires_grad=False, device='cpu'):
        self.device = device
        if device == 'gpu' and CUPY_AVAILABLE:
            self.data = cp.array(data) if not isinstance(data, cp.ndarray) else data
            self.xp = cp
        else:
            self.data = np.array(data) if not isinstance(data, np.ndarray) else data
            self.xp = np
            
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._prev = set()
        
    def backward(self, grad=None):
        if not self.requires_grad:
            return
            
        if grad is None:
            grad = self.xp.ones_like(self.data)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)
        
        build_topo(self)
        
        for node in reversed(topo):
            if node._grad_fn is not None:
                node._grad_fn()
    
    def zero_grad(self):
        self.grad = None
        for prev in self._prev:
            prev.zero_grad()
    
    def to(self, device):
        if device == self.device:
            return self
        if device == 'gpu' and CUPY_AVAILABLE:
            return Tensor(cp.array(self.data), self.requires_grad, device)
        else:
            data = self.data.get() if hasattr(self.data, 'get') else self.data
            return Tensor(np.array(data), self.requires_grad, 'cpu')
    
    def detach(self):
        return Tensor(self.data, requires_grad=False, device=self.device)
    
    def numpy(self):
        data = self.data.get() if hasattr(self.data, 'get') else self.data
        return np.array(data)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor({self.data}, device={self.device}, requires_grad={self.requires_grad})"
    
    # Operations
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    grad = out.grad
                    if self.shape != out.shape:
                        grad = self.xp.sum(grad, axis=tuple(range(grad.ndim - self.ndim)))
                        for i, (s1, s2) in enumerate(zip(self.shape, grad.shape)):
                            if s1 == 1 and s2 > 1:
                                grad = self.xp.sum(grad, axis=i, keepdims=True)
                    self.grad = grad if self.grad is None else self.grad + grad
                if other.requires_grad:
                    grad = out.grad
                    if other.shape != out.shape:
                        grad = self.xp.sum(grad, axis=tuple(range(grad.ndim - other.ndim)))
                        for i, (s1, s2) in enumerate(zip(other.shape, grad.shape)):
                            if s1 == 1 and s2 > 1:
                                grad = self.xp.sum(grad, axis=i, keepdims=True)
                    other.grad = grad if other.grad is None else other.grad + grad
            out._grad_fn = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    grad = out.grad * other.data
                    if self.shape != out.shape:
                        grad = self.xp.sum(grad, axis=tuple(range(grad.ndim - self.ndim)))
                        for i, (s1, s2) in enumerate(zip(self.shape, grad.shape)):
                            if s1 == 1 and s2 > 1:
                                grad = self.xp.sum(grad, axis=i, keepdims=True)
                    self.grad = grad if self.grad is None else self.grad + grad
                if other.requires_grad:
                    grad = out.grad * self.data
                    if other.shape != out.shape:
                        grad = self.xp.sum(grad, axis=tuple(range(grad.ndim - other.ndim)))
                        for i, (s1, s2) in enumerate(zip(other.shape, grad.shape)):
                            if s1 == 1 and s2 > 1:
                                grad = self.xp.sum(grad, axis=i, keepdims=True)
                    other.grad = grad if other.grad is None else other.grad + grad
            out._grad_fn = _backward
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if out.requires_grad:
            out._prev = {self, other}
            def _backward():
                if self.requires_grad:
                    grad = out.grad @ self.xp.swapaxes(other.data, -2, -1)
                    self.grad = grad if self.grad is None else self.grad + grad
                if other.requires_grad:
                    grad = self.xp.swapaxes(self.data, -2, -1) @ out.grad
                    other.grad = grad if other.grad is None else other.grad + grad
            out._grad_fn = _backward
        return out
    
    def __sub__(self, other):
        return self + (other * -1)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    grad = out.grad * power * (self.data ** (power - 1))
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = self.xp.zeros_like(self.data)
                    self.grad[idx] += out.grad
            out._grad_fn = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    grad = out.grad
                    if axis is not None and not keepdims:
                        grad = self.xp.expand_dims(grad, axis)
                    grad = self.xp.broadcast_to(grad, self.shape)
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    grad = out.grad.reshape(self.shape)
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if len(axes) == 0:
                        grad = out.grad.T
                    else:
                        inv_axes = self.xp.argsort(axes)
                        grad = out.grad.transpose(inv_axes)
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out
    
    def squeeze(self, axis=None):
        out = Tensor(self.xp.squeeze(self.data, axis=axis), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    grad = out.grad.reshape(self.shape)
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out
    
    def unsqueeze(self, axis):
        out = Tensor(self.xp.expand_dims(self.data, axis=axis), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    grad = out.grad.reshape(self.shape)
                    self.grad = grad if self.grad is None else self.grad + grad
            out._grad_fn = _backward
        return out

def cat(tensors, axis=0):
    """Concatenate tensors along an axis"""
    xp = tensors[0].xp
    device = tensors[0].device
    data = xp.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad, device=device)
    
    if out.requires_grad:
        out._prev = set(tensors)
        def _backward():
            indices = xp.cumsum(xp.array([t.shape[axis] for t in tensors[:-1]]))
            splits = xp.split(out.grad, indices, axis=axis)
            for t, grad in zip(tensors, splits):
                if t.requires_grad:
                    t.grad = grad if t.grad is None else t.grad + grad
        out._grad_fn = _backward
    return out

def stack(tensors, axis=0):
    """Stack tensors along a new axis"""
    xp = tensors[0].xp
    device = tensors[0].device
    data = xp.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires_grad, device=device)
    
    if out.requires_grad:
        out._prev = set(tensors)
        def _backward():
            grads = xp.split(out.grad, len(tensors), axis=axis)
            for t, grad in zip(tensors, grads):
                if t.requires_grad:
                    grad = xp.squeeze(grad, axis=axis)
                    t.grad = grad if t.grad is None else t.grad + grad
        out._grad_fn = _backward
    return out
