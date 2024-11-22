import os
import numpy as np
import matplotlib.pyplot as plt



def read_data(file_path):
    """
    Reads data from the specified file and returns it as a numpy array.
    Skips lines starting with #.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                data.append([float(value) for value in line.split()])
    return np.array(data)

def plot_columns(folder_paths, x_col, y_col, output_path, x_label='', y_label='', title='', font_size=12):
    """
    Reads data from phasefield_mbb_graphs.txt in the given folders and plots the specified columns.
    
    Parameters:
        folder_paths (list): List of folder paths containing phasefield_mbb_graphs.txt.
        x_col (int): Column index for the x-axis (0-based).
        y_col (int): Column index for the y-axis (0-based).
        output_path (str): File path to save the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        font_size (int): Font size for the labels and title.
    """
    plt.figure(figsize=(10, 6))
    
    for folder_path in folder_paths:
        file_path = os.path.join(folder_path, 'phasefield_mbb_graphs.txt')
        if os.path.exists(file_path):
            data = read_data(file_path)
            if x_col < data.shape[1] and y_col < data.shape[1]:
                label = os.path.basename(folder_path)
                plt.plot(data[:, x_col], data[:, y_col], label=label)
            else:
                print(f"Column indices out of range for file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

script_path = os.path.dirname(__file__)
output_file = os.path.join(script_path,"Ry_vs_uy_top.png")

# Example usage:
folders = [os.path.join(script_path,'E_var'), os.path.join(script_path,'E_high'), os.path.join(script_path,'E_min')]
plot_columns(
    folder_paths=folders,
    x_col=0,  # Time column
    y_col=2,  # Example: third data column
    output_path=output_file,
    x_label='u_y',
    y_label='Ry',
    title='',
    font_size=14
)
