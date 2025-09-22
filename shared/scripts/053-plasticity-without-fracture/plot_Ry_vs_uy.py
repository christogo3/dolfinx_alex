import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and convert data to a NumPy array
    data = [list(map(float, line.split())) for line in lines if not line.startswith('#')]
    data = np.array(data)
    return data[:, 2], data[:, 1]  # Return first column as x, second as y

def plot_single_file(file, x_label, y_label, legend, output_path):
    x, y = read_data(file)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=legend, linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.title("Scalar Quantities vs Time")
    plt.tight_layout()

    # Save to file instead of showing
    plt.savefig(output_path)
    plt.close()

# ==== USER SETTINGS ====
script_path = os.path.dirname(__file__)

file_path = os.path.join(script_path, 'run_simulation_graphs.txt')
output_file = os.path.join(script_path, 'Ry_vs_uy.png')

x_axis_label = 'Displacement (mm)'
y_axis_label = 'Reaction Force (N)'
legend_label = 'predictor corrector'

# ==== RUN THE PLOT ====
plot_single_file(file_path, x_axis_label, y_axis_label, legend_label, output_file)

