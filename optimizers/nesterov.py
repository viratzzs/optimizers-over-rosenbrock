from optimizers.base_optimizer import BaseOptimizer

class Nesterov(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta: float=0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = 0
        
    def step(self, params, grad):
        # TODO: implement lookahead grad calculation
        