import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class StochasticGradientDescent(BaseOptimizer):
    def __init__(self, lr: float=0.001):
        super().__init__(lr)
    
    def step(self, params, grad):
        noise = np.random.normal(0, 0.25) # adding high noise to kinda simulate sgd? don't like this at all tbh but ig my main focus is visualization here
        return (params - self.lr * (grad + noise))