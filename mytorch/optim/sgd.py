
# ============================================================================
# FILE: mytorch/optim/sgd.py
# ============================================================================
"""SGD Optimizer"""

class SGD:
    """Stochastic Gradient Descent with momentum"""
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [p.xp.zeros_like(p.data) for p in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                param.data -= self.lr * self.velocities[i]
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()