import numpy as np
import json
import os

from bgd import BatchGradientDescent
from base_optimizer import BaseOptimizer
from loss_functions import LossFunction, Rosenbrock

class SimulationVisualizer:
    """
    Simulates an optimization run over any loss function and visualizes it.
    """
    def __init__(self, optimizer: BaseOptimizer, loss_fn: LossFunction, init_pos: np.ndarray, steps: int):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.params = init_pos
        self.steps = steps
        self.trajectory = []
    
    def run(self):
        loss = self.loss_fn.evaluate(*self.params)
        self.trajectory.append((*self.params, loss))
        
        for _ in range(self.steps):
            #print(*self.params)
            grad = self.loss_fn.grad(*self.params)  # unpack array into positional args
            self.params = self.optimizer.step(self.params, grad) # update params with respective optimizer
            loss = self.loss_fn.evaluate(*self.params) # evaluate loss with updated params
            self.trajectory.append((*self.params, loss))

    def export_trajectory(self, path: str = "trajectories/gd.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(self.trajectory, open(path, "w"))

def main():
    #init_pos = np.array([-0.5, 2.0])
    init_pos = np.random.rand(2)
    #print(init_pos)
    print(f"Starting position: {init_pos}")
    s1 = SimulationVisualizer(BatchGradientDescent(), Rosenbrock(), init_pos, 10000)
    s1.run()
    s1.export_trajectory()
    
if __name__ == "__main__":
    main()