import math
import numpy as np

class LossFunction:
    """
    Represents a 2d/3d differentiable contour/surface.
    """
    def __init__(self):
        pass
    
    def evaluate(self, x: float, y: float) -> float:
        """Returns loss value for a point."""
        pass
    
    def grad(self, x: float, y: float, h: float = 1e-5) -> np.ndarray:
        """
        Returns gradient vector for a point. using Central Difference for approximation.
        """
        df_dx = (self.evaluate(x + h, y) - self.evaluate(x - h, y)) / (2 * h)
        df_dy = (self.evaluate(x, y + h) - self.evaluate(x, y - h)) / (2 * h)
        return np.array([df_dx, df_dy])

class Rosenbrock(LossFunction):
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b
        
    def evaluate(self, x, y):
        """Returns the value of expression ((a - x)**2 + b * (y - x**2)**2)"""
        return (self.a - x)**2 + (self.b * (y - x**2)**2) 

