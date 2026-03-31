

# ============================================================================
# FILE: mytorch/optim/rmsprop.py
# ============================================================================
"""RMSprop Optimizer"""

class RMSprop:
    """RMSprop optimizer"""
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v = [p.xp.zeros_like(p.data) for p in parameters]
        self.buffer = [p.xp.zeros_like(p.data) for p in parameters] if momentum > 0 else None
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
                
                if self.momentum > 0:
                    self.buffer[i] = self.momentum * self.buffer[i] + grad / (param.xp.sqrt(self.v[i]) + self.eps)
                    param.data -= self.lr * self.buffer[i]
                else:
                    param.data -= self.lr * grad / (param.xp.sqrt(self.v[i]) + self.eps)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()