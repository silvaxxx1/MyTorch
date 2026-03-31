
# ============================================================================
# FILE: mytorch/optim/lr_scheduler.py
# ============================================================================
"""Learning rate schedulers"""

import numpy as np

class _LRScheduler:
    """Base class for learning rate schedulers"""
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr
    
    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()
    
    def get_lr(self):
        raise NotImplementedError

class StepLR(_LRScheduler):
    """Decay learning rate by gamma every step_size epochs"""
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))

class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing learning rate schedule"""
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2

class OneCycleLR(_LRScheduler):
    """One cycle learning rate policy"""
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step_num = self.last_epoch
        if step_num < self.pct_start * self.total_steps:
            # Warmup
            return self.base_lr + (self.max_lr - self.base_lr) * step_num / (self.pct_start * self.total_steps)
        else:
            # Annealing
            progress = (step_num - self.pct_start * self.total_steps) / ((1 - self.pct_start) * self.total_steps)
            return self.max_lr * (1 - progress)