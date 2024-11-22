import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter


import alex.postprocessing as pp
import alex.homogenization as hom
import alex.linearelastic as le
import math

def plot_columns(data, col_x, col_y, output_filename, vlines=None, hlines=None, 
                 xlabel=None, ylabel=None, title=None, 
                 xlabel_fontsize=16, ylabel_fontsize=16, title_fontsize=18, 
                 tick_fontsize=14, figsize=(10, 6), usetex=False,
                 xlim=None, ylim=None):
    """
    Plot specified columns from the data with options for customizations and axis ranges.
    
    Parameters:
    - data: DataFrame containing the data to plot.
    - col_x: Column index for the x-axis.
    - col_y: Column index for the y-axis.
    - output_filename: Path to save the plot.
    - vlines: Optional list of x-coordinates for vertical lines.
    - hlines: Optional list of y-coordinates for horizontal lines.
    - xlabel, ylabel: Labels for x and y axes.
    - title: Plot title.
    - xlabel_fontsize, ylabel_fontsize, title_fontsize: Font sizes for labels and title.
    - tick_fontsize: Font size for axis tick labels.
    - figsize: Tuple for figure size.
    - usetex: Whether to use LaTeX rendering.
    - xlim: Tuple for x-axis range (e.g., (xmin, xmax)).
    - ylim: Tuple for y-axis range (e.g., (ymin, ymax)).
    """
    # Enable LaTeX rendering if requested
    if usetex:
        plt.rcParams['text.usetex'] = True

    # Set figure dimensions
    plt.figure(figsize=figsize)
    
    # Plot the data
    plt.plot(data[col_x], data[col_y], marker='o', linestyle='-')
    
    # Set custom labels and title, with specific font sizes
    plt.xlabel(xlabel if xlabel else f'Column {col_x}', fontsize=xlabel_fontsize)
    plt.ylabel(ylabel if ylabel else f'Column {col_y}', fontsize=ylabel_fontsize)
    plt.title(title if title else f' ', fontsize=title_fontsize)
    
    # Add dashed vertical lines if provided
    if vlines is not None:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
    
    # Add dashed horizontal lines if provided
    if hlines is not None:
        for hline in hlines:
            plt.axhline(y=hline, color='gray', linestyle='--', linewidth=1)
    
    # Set axis ranges if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    # Set the maximum number of ticks on each axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit x-axis to 10 ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit y-axis to 10 ticks
    
    # Set fontsize for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()  # Close the figure to prevent display in some environments
    print(f"Plot saved as {output_filename}")
    
    
def filter_data(data, col_index, lower_bound, upper_bound):
    """
    Filter rows based on whether the values in the specified column
    fall within the given range [lower_bound, upper_bound].

    :param data: DataFrame containing the data
    :param col_index: Index of the column to filter by
    :param lower_bound: Lower bound of the range (inclusive)
    :param upper_bound: Upper bound of the range (inclusive)
    :return: Filtered DataFrame
    """
    return data[(data[col_index] >= lower_bound) & (data[col_index] <= upper_bound)]


script_path = os.path.dirname(__file__)
simulation_data_folder = os.path.join(script_path,"simulation_20241122_135211")
data_path = os.path.join(simulation_data_folder, 'run_simulation_graphs.txt')

# Load the data from the text file, skipping the first row
data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)
data_filtered = filter_data(data,0,0.00000,100.0)

# Specify the output file path
output_file = os.path.join(script_path, 'xct_vs_t.png')
plot_columns(data_filtered, 0, 9, output_file,vlines=None,ylabel="$x_{ct} / L$",xlabel="$t / T$", usetex=False, title=" ")
