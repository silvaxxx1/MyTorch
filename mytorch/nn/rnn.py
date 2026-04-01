


# ============================================================================
# FILE: mytorch/nn/rnn.py
# ============================================================================
"""RNN layers"""

from ..tensor import Tensor, stack as tensor_stack
from .. import functional as F
from .modules import Module
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class RNNCell(Module):
    """Single RNN cell"""
    def __init__(self, input_size, hidden_size, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        scale = xp.sqrt(1.0 / hidden_size)
        self.W_ih = Tensor(xp.random.randn(input_size, hidden_size) * scale,
                          requires_grad=True, device=device)
        self.W_hh = Tensor(xp.random.randn(hidden_size, hidden_size) * scale,
                          requires_grad=True, device=device)
        self.bias = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        self._params = [self.W_ih, self.W_hh, self.bias]
    
    def __call__(self, x, h):
        return F.tanh(x @ self.W_ih + h @ self.W_hh + self.bias)

class RNN(Module):
    """Multi-layer RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = RNNCell(in_size, hidden_size, device)
            self.cells.append(cell)
            self._modules.append(cell)
    
    def __call__(self, x, h=None):
        xp = cp if self.device == 'gpu' and cp is not None else np
        seq_len, batch, _ = x.shape

        if h is None:
            h = [Tensor(xp.zeros((batch, self.hidden_size)), device=self.device)
                 for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # keeps grad connection via __getitem__
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
            outputs.append(x_t)

        return tensor_stack(outputs, axis=0), h

class GRUCell(Module):
    """Single GRU cell"""
    def __init__(self, input_size, hidden_size, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        scale = xp.sqrt(1.0 / hidden_size)
        
        # Reset gate
        self.W_ir = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hr = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_r = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        # Update gate
        self.W_iz = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hz = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_z = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        # New gate
        self.W_in = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hn = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_n = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        self._params = [self.W_ir, self.W_hr, self.b_r,
                       self.W_iz, self.W_hz, self.b_z,
                       self.W_in, self.W_hn, self.b_n]
    
    def __call__(self, x, h):
        r = F.sigmoid(x @ self.W_ir + h @ self.W_hr + self.b_r)
        z = F.sigmoid(x @ self.W_iz + h @ self.W_hz + self.b_z)
        n = F.tanh(x @ self.W_in + (r * h) @ self.W_hn + self.b_n)
        h_new = (Tensor(1, device=x.device) - z) * n + z * h
        return h_new

class GRU(Module):
    """Multi-layer GRU"""
    def __init__(self, input_size, hidden_size, num_layers=1, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = GRUCell(in_size, hidden_size, device)
            self.cells.append(cell)
            self._modules.append(cell)
    
    def __call__(self, x, h=None):
        xp = cp if self.device == 'gpu' and cp is not None else np
        seq_len, batch, _ = x.shape

        if h is None:
            h = [Tensor(xp.zeros((batch, self.hidden_size)), device=self.device)
                 for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
            outputs.append(x_t)

        return tensor_stack(outputs, axis=0), h

class LSTMCell(Module):
    """Single LSTM cell"""
    def __init__(self, input_size, hidden_size, device='cpu'):
        super().__init__()
        xp = cp if device == 'gpu' and cp is not None else np
        
        scale = xp.sqrt(1.0 / hidden_size)
        
        # Input gate
        self.W_ii = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hi = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_i = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        # Forget gate
        self.W_if = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hf = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_f = Tensor(xp.ones(hidden_size), requires_grad=True, device=device)
        
        # Cell gate
        self.W_ig = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_hg = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_g = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        # Output gate
        self.W_io = Tensor(xp.random.randn(input_size, hidden_size) * scale, requires_grad=True, device=device)
        self.W_ho = Tensor(xp.random.randn(hidden_size, hidden_size) * scale, requires_grad=True, device=device)
        self.b_o = Tensor(xp.zeros(hidden_size), requires_grad=True, device=device)
        
        self._params = [self.W_ii, self.W_hi, self.b_i,
                       self.W_if, self.W_hf, self.b_f,
                       self.W_ig, self.W_hg, self.b_g,
                       self.W_io, self.W_ho, self.b_o]
    
    def __call__(self, x, h, c):
        i = F.sigmoid(x @ self.W_ii + h @ self.W_hi + self.b_i)
        f = F.sigmoid(x @ self.W_if + h @ self.W_hf + self.b_f)
        g = F.tanh(x @ self.W_ig + h @ self.W_hg + self.b_g)
        o = F.sigmoid(x @ self.W_io + h @ self.W_ho + self.b_o)
        
        c_new = f * c + i * g
        h_new = o * F.tanh(c_new)
        
        return h_new, c_new

class LSTM(Module):
    """Multi-layer LSTM"""
    def __init__(self, input_size, hidden_size, num_layers=1, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(in_size, hidden_size, device)
            self.cells.append(cell)
            self._modules.append(cell)
    
    def __call__(self, x, h=None, c=None):
        xp = cp if self.device == 'gpu' and cp is not None else np
        seq_len, batch, _ = x.shape

        if h is None:
            h = [Tensor(xp.zeros((batch, self.hidden_size)), device=self.device)
                 for _ in range(self.num_layers)]
        if c is None:
            c = [Tensor(xp.zeros((batch, self.hidden_size)), device=self.device)
                 for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, h[i], c[i])
                x_t = h[i]
            outputs.append(x_t)

        return tensor_stack(outputs, axis=0), h, c