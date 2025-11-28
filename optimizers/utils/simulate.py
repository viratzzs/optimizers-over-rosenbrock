import numpy as np
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from optimizers.bgd import *
from optimizers.sgd import *
from optimizers.momentum import Momentum
from optimizers.adam import Adam
from optimizers.base_optimizer import BaseOptimizer
from optimizers.loss_functions import LossFunction, Rosenbrock

class Simulator:
    def __init__(self, optimizer: BaseOptimizer, loss_fn: LossFunction, params: np.ndarray, steps: int):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.params = params
        self.steps = steps
        self.trajectory = []
    
    def run(self):
        # eval initial loss
        loss = self.loss_fn.evaluate(self.params)
        self.trajectory.append((self.params.tolist(), loss.tolist() if isinstance(loss, np.ndarray) else loss))
        
        for _ in range(self.steps):
        #    grad = self.loss_fn.grad(*self.params)  # unpack array into positional args
            grad = self.loss_fn.grad(self.params)
            if isinstance(self.optimizer, Adam):
                self.params = self.optimizer.step(self.params, grad, cur_step=_ + 1)
            else:
                self.params = self.optimizer.step(self.params, grad) # update params w/ selected optimizer
            loss = self.loss_fn.evaluate(self.params) # calculate loss
        #    self.trajectory.append((*self.params, loss))
            self.trajectory.append((self.params.tolist(), loss.tolist() if isinstance(loss, np.ndarray) else loss))
        
        print(self.trajectory[-10:])
        print(np.array(self.trajectory, dtype=object).shape)

    def export_trajectory(self, path: str = "trajectories/gd.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.trajectory, open(path, "w"))

def main():
    #init_pos = np.array([-0.5, 2.0])
    #init_pos = np.random.rand(2)
    params = np.random.rand(2,100)
    print(f"Starting position: {params}")
    #s1 = Simulator(BatchGradientDescent(), Rosenbrock(params.shape, baby_mode=False), params, 7500)
    #s1 = Simulator(StochasticGradientDescent(), Rosenbrock(params.shape, baby_mode=False), params, 7500)
    #s1 = Simulator(Momentum(), Rosenbrock(params.shape, baby_mode=False), params, 7500)
    s1 = Simulator(Adam(), Rosenbrock(params.shape, baby_mode=False), params, 7500)
    s1.run()
    #s1.export_trajectory()
    
if __name__ == "__main__":
    main()