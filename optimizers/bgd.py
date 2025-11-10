from base_optimizer import BaseOptimizer

class BatchGradientDescent(BaseOptimizer):
    """
    Implements Batch Gradient Descent, looping over all training samples as a batch and then averaging the gradients.
    """
    def __init__(self, lr: float = 0.001):
        super().__init__(lr)
    
    def step(self, params, grad):
        return params - self.lr * grad
