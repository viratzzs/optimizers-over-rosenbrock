import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class Adagrad(BaseOptimizer):
    def __init__(self, lr: float=0.0001):
        super().__init__(lr)
        self.cache = 0
        
    def step(self, params, grad):
        self.cache += grad**2
        new_params = params - (self.lr / np.sqrt(self.cache + 1e-8)) * grad # add epsilon
        return new_params