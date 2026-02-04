from manim import *
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from optimizers.utils.simulate import Simulator
from optimizers import sgd, bgd
from optimizers.adam import Adam
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.adagrad import Adagrad 
from optimizers.adadelta import Adadelta    
from optimizers.rmsprop import RMSprop
from optimizers.adamax import AdaMax
from optimizers.nadam import Nadam 
from optimizers.loss_functions import Rosenbrock

# label axis points
class Optimizer2D(Scene):
    def construct(self):
        #params = np.random.rand(2) * 4 - 2
        params = np.random.rand(2, 200) * 4 - 2
        print(f"Initial weights: {params}")
        
        # lower lr for more params
        #sim = Simulator(Nadam(lr=0.001), Rosenbrock(shape=params.shape, baby_mode=False), params, 10000)
        sim = Simulator(AdaMax(lr=0.001), Rosenbrock(shape=params.shape, baby_mode=False), params, 15000)
        #sim = Simulator(Nadam(lr=0.0001), Rosenbrock(shape=params.shape, baby_mode=False), params, 10000)
        sim.run()
        data = sim.trajectory
        
        def f(x, y):
            a, b = 1, 100
            return (a - x)**2 + b * (y - x**2)**2

        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=10,
            axis_config={"color": GREY},
            tips=False
        )
        # won't run due to some miktex build error
        #axes.add_coordinates()
        x_coords = VGroup()
        for i in range(-1, 2):
            label = Text(str(i), font_size=14, color=WHITE).next_to(axes.c2p(i, 0), DOWN, buff=0.2)
            x_coords.add(label)

        y_coords = VGroup()
        for i in range(-1, 2):
            if i != 0:
                label = Text(str(i), font_size=14, color=WHITE).next_to(axes.c2p(0, i), LEFT, buff=0.2)
                y_coords.add(label)

        self.add(x_coords, y_coords)
        
        x_vals = np.linspace(-2, 2, 800)
        y_vals = np.linspace(-2, 2, 800)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f(X, Y)
        
        levels = np.linspace(Z.min(), Z.max(), 150)
        
        fig_temp, ax_temp = plt.subplots()
        cs = ax_temp.contour(X, Y, Z, levels=levels)
        plt.close(fig_temp)
        
        contours = VGroup()
        
        for level_segments in cs.allsegs:
            for segment in level_segments:
                if len(segment) > 1:
                    # Convert to manim coords
                    manim_points = [axes.c2p(v[0], v[1]) for v in segment]
                    contour_line = VMobject()
                    contour_line.set_points_smoothly(manim_points)
                    contour_line.set_stroke(BLUE, width=1.5, opacity=0.5)
                    contours.add(contour_line)
        
        min_point = Dot(axes.c2p(1, 1), color=GREEN, radius=0.1)
        min_label = Text("minima", font_size=20, color=GREEN).next_to(min_point, RIGHT, buff=0.15)
        
        origin = Dot(axes.c2p(0, 0), color=WHITE, radius=0.06)
        origin_label = Text("(0,0)", font_size=16, color=WHITE).next_to(origin, DOWN+RIGHT, buff=0.1)
        
        self.add(axes, contours, min_point, min_label, origin, origin_label)
        
        # Calculate mean position(centroid) of tensor at each step
        # d[0] is array of shape (2, N)
        # We want (mean_x, mean_y)
        path_points = []
        for d in data:
            params = np.array(d[0])
            if params.ndim > 1:
                mean_x = np.mean(params[0])
                mean_y = np.mean(params[1])
                
                # Stop plotting if values explode to NaN/Inf
                if not (np.isfinite(mean_x) and np.isfinite(mean_y)):
                    break
                    
                path_points.append(axes.c2p(mean_x, mean_y))
            else:
                path_points.append(axes.c2p(params[0], params[1]))
        
        path = VMobject(color=BLUE, stroke_width=3)
        path.set_points_as_corners(path_points)
        
        dot = Dot(path_points[0], color=RED, radius=0.1)
        
        start_label = Text("Start", font_size=18, color=WHITE).next_to(path_points[0], DOWN, buff=0.15)
        end_label = Text("End", font_size=18, color=WHITE).next_to(path_points[-1], UP, buff=0.15)
        self.add(start_label, dot)
        
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=8,
            rate_func=linear
        )
        
        self.play(FadeIn(end_label), run_time=0.5)
        self.wait(2)
