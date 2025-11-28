from optimizers.base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta1: float=0.9, beta2: float=0.999):
        super().__init__(lr)
        self.momentum = 0 # first moment, mean
        self.velocity = 0 # second moment, uncentered variance
        self.beta1 = beta1
        self.beta2 = beta2
        
    def step(self, params, grad, cur_step):
        self.momentum = (self.beta1 * self.momentum) + (1 - self.beta1) * grad
        self.velocity = (self.beta2 * self.velocity) + (1 - self.beta2) * grad ** 2
        
        # correcting bias in first and second moments
        mom1 = self.momentum / (1 - self.beta1 ** cur_step)
        mom2 = self.velocity / (1 - self.beta2 ** cur_step)
        
        new_params = params - (self.lr * mom1 / (mom2 ** 0.5 + 1e-8))
        
        return new_params