import math
import numpy as np

class LossFunction:
    """
    Represents a 2d/3d differentiable contour/surface.
    """
    def __init__(self):
        pass
    
    def evaluate(self, params: np.ndarray) -> float:
        """Returns loss value for a point."""
        pass
    
    #def grad(self, x: float, y: float, h: float = 1e-5) -> np.ndarray:
        #df_dx = (self.evaluate(x + h, y) - self.evaluate(x - h, y)) / (2 * h)
        #df_dy = (self.evaluate(x, y + h) - self.evaluate(x, y - h)) / (2 * h)
        #return np.array([df_dx, df_dy])
    def grad(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Returns gradient vector for a point. using Central Difference for approximation.
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            
            p_plus[i] += h
            p_minus[i] -= h
            
            grad[i] = (self.evaluate(p_plus) - self.evaluate(p_minus)) / (2 * h)
        return grad

class Rosenbrock(LossFunction):
    def __init__(self, shape: tuple = None, a: float = 1.0, b: float = 100.0, baby_mode: str = True):
        super().__init__()
        if baby_mode:
            self.a = a
            self.b = b
        else:
            self.a = np.full((1, shape[-1]), a) # (1,20)
            self.b = np.full((1, shape[-1]), b) # same
        
    def evaluate(self, params):
        x, y = params[0], params[1]
        return (self.a - x)**2 + (self.b * (y - x**2)**2) 

