

# ============================================================================
# FILE: mytorch/utils/early_stopping.py
# ============================================================================
"""Early stopping utility"""

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.monitor_op = lambda x, y: x > y + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop