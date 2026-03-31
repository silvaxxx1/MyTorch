
# ============================================================================
# FILE: mytorch/utils/model_utils.py
# ============================================================================
"""Model utility functions"""

import pickle
import numpy as np

def save_checkpoint(model, optimizer, epoch, filepath, **kwargs):
    """Save model checkpoint"""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': {
            'lr': optimizer.lr,
            't': getattr(optimizer, 't', 0)
        },
        'epoch': epoch,
        **kwargs
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filepath, model=None, optimizer=None):
    """Load model checkpoint"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.lr = checkpoint['optimizer_state']['lr']
        if 't' in checkpoint['optimizer_state']:
            optimizer.t = checkpoint['optimizer_state']['t']
    
    return checkpoint

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.data.size for p in model.parameters())

def model_summary(model, input_shape):
    """Print model summary"""
    print("=" * 80)
    print(f"{'Layer':<30} {'Output Shape':<20} {'Param #':<15}")
    print("=" * 80)
    
    total_params = 0
    for i, module in enumerate(model._modules if hasattr(model, '_modules') else [model]):
        params = sum(p.data.size for p in module.parameters())
        total_params += params
        print(f"{module.__class__.__name__:<30} {'N/A':<20} {params:<15,}")
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print("=" * 80)