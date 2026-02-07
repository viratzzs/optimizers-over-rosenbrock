import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from optimizers.utils.simulate import Simulator
from optimizers.adam import Adam
from optimizers.adamax import AdaMax
from optimizers.nadam import Nadam
from optimizers.rmsprop import RMSprop
from optimizers.adadelta import Adadelta
from optimizers.adagrad import Adagrad
from optimizers.momentum import Momentum
from optimizers.nesterov import Nesterov
from optimizers.loss_functions import Rosenbrock


def visualize_optimizer_3d_animated(optimizers, loss_function, num_steps, plot_title, output_file, fps, skip_frames):
    colors = ['#00ffff', '#ff69b4', '#32cd32', '#ffd700', '#ff8c00', '#9d4edd', "#fbfbfb", '#ff3333']
    
    all_paths = []
    x_min_global, x_max_global = float('inf'), float('-inf')
    y_min_global, y_max_global = float('inf'), float('-inf')
    
    for i, (optimizer, name) in enumerate(optimizers):
        opt_initial_params = initial_params.copy()
        
        sim = Simulator(optimizer, loss_function, opt_initial_params, num_steps)
        trajectory = sim.run()
        
        x_path = []
        y_path = []
        z_path = []
        
        for params, loss in trajectory:
            params_array = np.array(params)
            x_path.append(float(params_array[0][0]) if params_array[0].size > 0 else float(params_array[0]))
            y_path.append(float(params_array[1][0]) if params_array[1].size > 0 else float(params_array[1]))
            z_path.append(float(np.mean(loss)) if isinstance(loss, (np.ndarray, list)) else float(loss))
        
        x_path = np.array(x_path)
        y_path = np.array(y_path)
        z_path = np.array(z_path)
        
        all_paths.append((x_path, y_path, z_path, name))
        
        x_min_global = min(x_min_global, x_path.min())
        x_max_global = max(x_max_global, x_path.max())
        y_min_global = min(y_min_global, y_path.min())
        y_max_global = max(y_max_global, y_path.max())
    
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -2.0, 2.0
    
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            params_temp = np.array([[X[i, j]], [Y[i, j]]])
            loss_val = loss_function.evaluate(params_temp)
            Z[i, j] = float(np.mean(loss_val)) if isinstance(loss_val, (np.ndarray, list)) else loss_val
    
    fig = plt.figure(figsize=(14, 10), dpi=100, facecolor='#1a1a1a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a1a')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.twilight, alpha=0.4, 
                            edgecolor='none', antialiased=True, 
                            vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 95))
    
    ax.set_xlabel('Parameter 1 (x)', fontsize=12, labelpad=10, color='white')
    ax.set_ylabel('Parameter 2 (y)', fontsize=12, labelpad=10, color='white')
    ax.set_zlabel('Loss', fontsize=12, labelpad=10, color='white')
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20, color='white')
    
    ax.tick_params(colors='white')
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.2, color='gray')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    lines = []
    current_points = []
    for i, (x_path, y_path, z_path, name) in enumerate(all_paths):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], [], color, linewidth=3, label=name, zorder=10)
        lines.append(line)
        point = ax.scatter([], [], [], color=color, s=150, marker='o', 
                            edgecolors='white', linewidths=2, zorder=12)
        current_points.append(point)
    
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    legend.get_frame().set_facecolor('#1a1a1a')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines + current_points
    
    max_path_len = max(len(path[0]) for path in all_paths)
    num_frames = max_path_len // skip_frames + 1
    
    def update(frame):
        idx = frame * skip_frames
        
        for i, (x_path, y_path, z_path, name) in enumerate(all_paths):
            actual_idx = min(idx, len(x_path) - 1)
            lines[i].set_data(x_path[:actual_idx+1], y_path[:actual_idx+1])
            lines[i].set_3d_properties(z_path[:actual_idx+1])
            current_points[i]._offsets3d = ([x_path[actual_idx]], [y_path[actual_idx]], [z_path[actual_idx]])
        
        azim = 45 + (frame * 22.5 / num_frames)
        ax.view_init(elev=25, azim=azim)
        
        return lines + current_points
    
    print(f"Creating animation with {num_frames} frames...")
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, 
                            interval=1000//fps, blit=False, repeat=True)
    
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_file, writer=writer)
    
    print(f"\nOptimization Statistics for all optimizers:")
    for i, (x_path, y_path, z_path, name) in enumerate(all_paths):
        print(f"\n{name}:")
        print(f"  Initial loss: {z_path[0]:.6f}")
        print(f"  Final loss: {z_path[-1]:.6f}")
        print(f"  Loss reduction: {z_path[0] - z_path[-1]:.6f}")
        print(f"  Final parameters: x={x_path[-1]:.6f}, y={y_path[-1]:.6f}")
    
    print(f"\nAnimation saved as '{output_file}'")
    
    return fig, ax, all_paths


if __name__ == "__main__":
    np.random.seed(9999)
    
    initial_params = np.random.rand(2, 1) * 4 - 2
    print(f"Initial parameters: {initial_params.flatten()}")
    
    loss_fn = Rosenbrock(shape=initial_params.shape, baby_mode=False)
    
    optimizers = [
        (Adam(lr=0.001), "Adam"),
        (AdaMax(lr=0.001), "AdaMax"),
        (Nadam(lr=0.001), "Nadam"),
        (RMSprop(lr=0.001), "RMSprop"),
        (Adadelta(lr=0.001), "Adadelta"),
        (Adagrad(lr=0.001), "Adagrad"),
        (Momentum(lr=0.0001), "Momentum"),
        (Nesterov(lr=0.0001), "Nesterov"),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running all optimizers over the loss curve")
    print('='*60)
    
    fig, ax, paths = visualize_optimizer_3d_animated(
        optimizers=optimizers,
        loss_function=loss_fn,
        num_steps=7500,
        plot_title="Optimization over Rosenbrock",
        output_file="all_optimizers_3d.mp4",
        fps=24,
        skip_frames=20
    )
