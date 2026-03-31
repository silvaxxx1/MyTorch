

# ============================================================================
# FILE: mytorch/utils/clip_grad.py
# ============================================================================
"""Gradient clipping utilities"""

import numpy as np

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """Clip gradient norm of parameters"""
    if norm_type == float('inf'):
        total_norm = max(p.grad.max() if p.grad is not None else 0 for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = (p.xp.abs(p.grad) ** norm_type).sum()
                total_norm += param_norm
        total_norm = total_norm ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef
    
    return total_norm

def clip_grad_value_(parameters, clip_value):
    """Clip gradient values of parameters"""
    for p in parameters:
        if p.grad is not None:
            p.grad = p.xp.clip(p.grad, -clip_value, clip_value)
