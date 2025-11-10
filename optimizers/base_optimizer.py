class BaseOptimizer:
    """
    Abstract base class for all optimizers.
    """
    def __init__(self, lr: float):
        self.lr = lr
    
    def step(self, grad: tuple[float, float]) -> tuple[float, float]:
        pass
    
    def reset(self):
        pass
    
    def log_step(self, position: tuple[float, float], loss: float):
        pass
