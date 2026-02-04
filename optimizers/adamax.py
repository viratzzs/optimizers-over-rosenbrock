import numpy as np
from optimizers.base_optimizer import BaseOptimizer

class AdaMax(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta1: float=0.9, beta2: float=0.999):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.moment1 = 0 # momentum
        self.moment2 = 0 # infinity norm which converges to a more stable value, hence the max()

    def step(self, params, grad, cur_step):
        # legit adam but the second moment's norm is approximated to the infinity
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * grad
        self.moment2 = np.maximum(self.beta2 * self.moment2, np.abs(grad))
        
        new_params = params - (self.lr * self.moment1) / (self.moment2 * (1 - self.beta1**cur_step)) # dividing by bias correction term for first moment
        return new_params