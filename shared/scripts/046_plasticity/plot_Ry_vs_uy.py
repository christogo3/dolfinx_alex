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

def plot_two_files(file1, file2, x_label, y_label, legend1, legend2, output_path):
    x1, y1 = read_data(file1)
    x2, y2 = read_data(file2)

    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label=legend1, linewidth=2)
    plt.plot(x2, y2, label=legend2, linewidth=2)

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

file_path1 = os.path.join(script_path, 'run_simulation_loading_unloading_without_fracture_graphs.txt')   # e.g., os.path.join('data', 'data1.txt')
file_path2 = os.path.join(script_path, '..', '043_ramberg_osgood', 'run_simulation_loading_unloading_without_fracture_graphs.txt')  # e.g., os.path.join('data', 'data2.txt')
output_file = os.path.join(script_path, 'Ry_vs_uy.png')      # e.g., os.path.join('plots', 'result.png')

x_axis_label = 'Displacement (mm)'
y_axis_label = 'Reaction Force (N)'
legend_label1 = 'predictor corrector'
legend_label2 = 'ramberg osgood'

# ==== RUN THE PLOT ====
plot_two_files(file_path1, file_path2, x_axis_label, y_axis_label, legend_label1, legend_label2, output_file)
