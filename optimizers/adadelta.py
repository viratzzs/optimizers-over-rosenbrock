from optimizers.base_optimizer import BaseOptimizer

class Adadelta(BaseOptimizer):
    def __init__(self, lr: float=0.0001):
        super().__init__(lr)
        