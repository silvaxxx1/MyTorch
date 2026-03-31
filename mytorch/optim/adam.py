
# ============================================================================
# FILE: mytorch/optim/adam.py
# ============================================================================
"""Adam Optimizer"""

class Adam:
    """Adam optimizer"""
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [p.xp.zeros_like(p.data) for p in parameters]
        self.v = [p.xp.zeros_like(p.data) for p in parameters]
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param.data -= self.lr * m_hat / (param.xp.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()