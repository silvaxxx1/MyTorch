

# ============================================================================
# FILE: mytorch/nn/attention.py
# ============================================================================
"""Attention mechanisms"""

from ..tensor import Tensor
from .. import functional as F
from .modules import Module
from .linear import Linear
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class MultiHeadAttention(Module):
    """Multi-Head Attention for Transformers"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, device='cpu'):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = Linear(embed_dim, embed_dim, device=device)
        self.k_proj = Linear(embed_dim, embed_dim, device=device)
        self.v_proj = Linear(embed_dim, embed_dim, device=device)
        self.out_proj = Linear(embed_dim, embed_dim, device=device)
        
        self._modules = [self.q_proj, self.k_proj, self.v_proj, self.out_proj]
    
    def __call__(self, query, key, value, mask=None):
        batch, seq_len, _ = query.shape
        
        q = self.q_proj(query).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)
