"""MyTorch: A minimal deep learning framework inspired by PyTorch"""

from .tensor import Tensor
from . import nn
from . import optim
from . import functional as F
from . import data
from . import utils

__version__ = "0.1.0"

def manual_seed(seed):
    """Set random seed for reproducibility"""
    import numpy as np
    np.random.seed(seed)
    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass