from manim import *
import json
import matplotlib.pyplot as plt

class Optimizer2D(Scene):
    def construct(self):
        with open("trajectories/gd.json") as f:
            data = json.load(f)
        
        def f(x, y):
            a, b = 1, 100
            return (a - x)**2 + b * (y - x**2)**2

        # Create axes showing the full Rosenbrock banana valley
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 4, 1],
            x_length=8,
            y_length=12,
            axis_config={"color": GREY},
            tips=False
        )
        
        # Generate contour data using matplotlib
        x_vals = np.linspace(-2, 2, 300)
        y_vals = np.linspace(-2, 4, 300)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f(X, Y)
        
        # Create contour levels using quantiles for even visual spacing
        levels = np.unique(np.quantile(Z.flatten(), np.linspace(0.02, 0.98, 36)))
        
        # Use matplotlib to generate contour paths
        fig_temp, ax_temp = plt.subplots()
        cs = ax_temp.contour(X, Y, Z, levels=levels)
        plt.close(fig_temp)
        
        # Extract contour lines and convert to Manim objects
        contours = VGroup()
        
        # cs.allsegs is a list of lists: allsegs[level_idx][segment_idx] contains vertices
        for level_segments in cs.allsegs:
            for segment in level_segments:
                if len(segment) > 1:
                    # Convert to Manim coordinates
                    manim_points = [axes.c2p(v[0], v[1]) for v in segment]
                    contour_line = VMobject()
                    contour_line.set_points_as_corners(manim_points)
                    contour_line.set_stroke(BLUE, width=1.5, opacity=0.5)
                    contours.add(contour_line)
        
        # Mark the global minimum at (1, 1)
        min_point = Dot(axes.c2p(1, 1), color=GREEN, radius=0.1)
        min_label = Text("min (1,1)", font_size=20, color=GREEN).next_to(min_point, RIGHT, buff=0.15)
        
        # Add reference dots to verify coordinate system
        origin = Dot(axes.c2p(0, 0), color=WHITE, radius=0.06)
        origin_label = Text("(0,0)", font_size=16, color=WHITE).next_to(origin, DOWN+RIGHT, buff=0.1)
        
        # Add everything to the scene
        self.add(axes, contours, min_point, min_label, origin, origin_label)
        
        # Convert trajectory to Manim coordinates
        path_points = [axes.c2p(d[0], d[1]) for d in data]
        
        # Create the path
        path = VMobject(color=BLUE, stroke_width=3)
        path.set_points_as_corners(path_points)
        
        # Create a moving dot
        dot = Dot(path_points[0], color=RED, radius=0.1)
        
        # Add start label
        start_label = Text("Start", font_size=18, color=WHITE).next_to(path_points[0], DOWN, buff=0.15)
        end_label = Text("End", font_size=18, color=WHITE).next_to(path_points[-1], UP, buff=0.15)
        self.add(start_label, dot)
        
        # Animate the path and dot
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=8,
            rate_func=linear
        )
        
        # Show end label after path completes
        self.play(FadeIn(end_label), run_time=0.5)
        self.wait(2)
    
class Optimizer3D(Scene):
    def construct(self):
        pass