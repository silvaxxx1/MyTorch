

# ============================================================================
# FILE: mytorch/nn/transformer.py
# ============================================================================
"""Transformer layers"""

from .modules import Module
from .linear import Linear
from .attention import MultiHeadAttention
from .normalization import LayerNorm
from .dropout import Dropout
from .. import functional as F

class TransformerEncoderLayer(Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, device)
        self.linear1 = Linear(d_model, dim_feedforward, device=device)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, device=device)
        
        self.norm1 = LayerNorm(d_model, device=device)
        self.norm2 = LayerNorm(d_model, device=device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self._modules = [self.self_attn, self.linear1, self.linear2, self.norm1, self.norm2]
    
    def __call__(self, src, src_mask=None):
        # Self attention
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderLayer(Module):
    """Transformer Decoder Layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, device)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout, device)
        self.linear1 = Linear(d_model, dim_feedforward, device=device)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, device=device)
        
        self.norm1 = LayerNorm(d_model, device=device)
        self.norm2 = LayerNorm(d_model, device=device)
        self.norm3 = LayerNorm(d_model, device=device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self._modules = [self.self_attn, self.multihead_attn, self.linear1, 
                        self.linear2, self.norm1, self.norm2, self.norm3]
    
    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
