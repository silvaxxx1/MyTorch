
# ============================================================================
# FILE: mytorch/nn/modules.py
# ============================================================================
"""Base module class"""

class Module:
    """Base class for all layers"""
    def __init__(self):
        self._params = []
        self._modules = []
        self.training = True
    
    def parameters(self):
        params = list(self._params)
        for module in self._modules:
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def train(self):
        self.training = True
        for module in self._modules:
            module.train()
    
    def eval(self):
        self.training = False
        for module in self._modules:
            module.eval()
    
    def to(self, device):
        for i, p in enumerate(self._params):
            self._params[i] = p.to(device)
        for module in self._modules:
            module.to(device)
        return self
    
    def state_dict(self):
        """Return state dictionary"""
        return {'params': [p.numpy() for p in self.parameters()]}
    
    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        params = self.parameters()
        for i, p_data in enumerate(state_dict['params']):
            if i < len(params):
                params[i].data = params[i].xp.array(p_data)

class Sequential(Module):
    """Sequential container"""
    def __init__(self, *layers):
        super().__init__()
        self._modules = list(layers)
    
    def __call__(self, x):
        for layer in self._modules:
            x = layer(x)
        return x
