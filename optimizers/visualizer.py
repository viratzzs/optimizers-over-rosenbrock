from manim import *
import json
import matplotlib.pyplot as plt
import numpy as np
from simulate import SimulationVisualizer
from bgd import BatchGradientDescent
from loss_functions import Rosenbrock

class Optimizer2D(Scene):
    def construct(self):
        # Generate a fresh trajectory with random starting point
        init_pos = np.random.rand(2) * 4 - 2  # Random point in [-2, 2] range
        print(f"Starting position: {init_pos}")
        
        sim = SimulationVisualizer(BatchGradientDescent(), Rosenbrock(), init_pos, 10000)
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
                    # Convert to manim coordinates
                    manim_points = [axes.c2p(v[0], v[1]) for v in segment]
                    contour_line = VMobject()
                    contour_line.set_points_smoothly(manim_points)  
                    contour_line.set_stroke(BLUE, width=1.5, opacity=0.5)
                    contours.add(contour_line)
        
        min_point = Dot(axes.c2p(1, 1), color=GREEN, radius=0.1)
        min_label = Text("min (1,1)", font_size=20, color=GREEN).next_to(min_point, RIGHT, buff=0.15)
        
        origin = Dot(axes.c2p(0, 0), color=WHITE, radius=0.06)
        origin_label = Text("(0,0)", font_size=16, color=WHITE).next_to(origin, DOWN+RIGHT, buff=0.1)
        
        self.add(axes, contours, min_point, min_label, origin, origin_label)
        
        path_points = [axes.c2p(d[0], d[1]) for d in data]
        
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
    
#class Optimizer3D(Scene):
#    def construct(self):
#        pass