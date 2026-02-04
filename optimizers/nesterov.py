from optimizers.base_optimizer import BaseOptimizer

class Nesterov(BaseOptimizer):
    def __init__(self, lr: float=0.0001, beta: float=0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = 0
        
    def step(self, params, grad):
        # DONE: implement lookahead grad calculation by ilya's formulation
        self.velocity = self.beta * self.velocity + grad # adding raw gradients w/o lr
        new_params = params - self.lr * (grad + self.beta * self.velocity) # now applying lr to current gradients AND updated velocity
        
        return new_params