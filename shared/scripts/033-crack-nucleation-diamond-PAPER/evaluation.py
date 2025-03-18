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
import alex.phasefield as pf
import alex.homogenization as hom
import alex.linearelastic as le
import alex.evaluation as ev
import math

from scipy.signal import savgol_filter
import numpy as np


import seaborn as sns
from matplotlib.lines import Line2D

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

def remove_first_percentage(data, percentage):
    """
    Remove the first 'x%' of rows from the DataFrame.

    :param data: DataFrame containing the data
    :param percentage: Percentage of rows to remove from the top (0 to 100)
    :return: DataFrame with the first 'x%' of rows removed
    """
    # Calculate the number of rows to remove
    rows_to_remove = int(len(data) * (percentage / 100))
    
    # Return the DataFrame excluding the first 'x%' rows
    return data.iloc[rows_to_remove:].reset_index(drop=True)

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def find_max_y_under_x_threshold(df, x_col, y_col, x_threshold):
    """
    Filters the points where the x-value is smaller than the given threshold
    and returns the point with the maximum y-value from this set.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - x_col (str or int): Column name or index for the x-coordinates.
    - y_col (str or int): Column name or index for the y-coordinates.
    - x_threshold (float): The x-value threshold to filter points.
    
    Returns:
    - (float, float): The (x, y) coordinates of the point with the maximum y-value
      among those with x < x_threshold.
    """
    # Filter points where x < x_threshold
    filtered_df = df[df[x_col] < x_threshold]
    
    # Check if the filtered set is not empty
    if filtered_df.empty:
        raise ValueError(f"No points found with x < {x_threshold}")
    
    # Find the index of the maximum y-value among the filtered points
    max_y_index = filtered_df[y_col].idxmax()
    
    # Return the (x, y) values of the point with the maximum y
    max_y_point = filtered_df.loc[max_y_index, [x_col, y_col]]
    
    return max_y_point[x_col], max_y_point[y_col]

# Needs to be set
DHOLE = 1.0
init_crack = DHOLE/2.0

col_crack_length = 9
label_crack_length = "$A / L$"

diamond_folder_name = "033-crack-nucleation-diamond-PAPER"
circular_holes_folder_name = "032-crack-nucleation-2D-holes-PAPER"

height_label = "$h_d$"
width_label = "$w_d$"
initial_crack_label = "$\Delta{a}$"
circular_label = "circular hole"
diamond_label = "quadratic hole"
velocity_bc_label = "$\dot{x}^{bc}$"


# col_crack_length = 3
# label_crack_length = "$x_{ct} / L$"


script_path = os.path.dirname(__file__)
# Define the height_study folder path
height_study_folder = os.path.join(script_path, "height_study")

# Get a list of all folders in the height_study folder
all_folders = [f for f in os.listdir(height_study_folder) if os.path.isdir(os.path.join(height_study_folder, f))]

# Find the first folder that starts with "simulation_"
simulation_folder = next((f for f in all_folders if f.startswith("simulation_")), None)

if simulation_folder:
    # Construct the full path to the simulation folder
    simulation_data_folder = os.path.join(height_study_folder, simulation_folder)
    print(f"Selected simulation folder: {simulation_data_folder}")
else:
    print("No folder starting with 'simulation_' found.")
data_path = os.path.join(simulation_data_folder, 'run_simulation_K_sym_graphs.txt')
# data_directory = "./"

# Load the data from the text file, skipping the first row
data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)
data_filtered = filter_data(data,0,0.00000,100.0)

# Specify the output file path
output_file = os.path.join(script_path, 'xct_vs_t.png')
plot_columns(data_filtered, 0, 3, output_file,vlines=None,ylabel="$x_{ct} / L$",xlabel="$t / T$", usetex=False, title=" ")

simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"height_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
height_values = []
initial_crack_length_values = []



def filter_data_to_range_where_A_exceeds_initial_value(filter_data, initial_crack_param, data, param):
    eps=param["eps"]
    e0=param["e0"]
    # A_initial_num = (1.0 + e0 / eps) * initial_crack_param / 2.0  # divided by two since symmetric computation
    A_initial_num = (1.0 + e0 / eps) * initial_crack_param / 2.0
    
    data = filter_data(data,col_crack_length,A_initial_num * 1.8,param["width"])
    
    t_start_brutal_crack_growth = data.values[0][0]
    Delta_t_initial_crack_growth = 0.05
    return data,A_initial_num,t_start_brutal_crack_growth,Delta_t_initial_crack_growth

for sim in simulation_results:
    
    param = sim[1]
    data = sim[0]
    
    # filter data so data row starts with initial brutal crack growth
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    dhole = param["dhole"]
    height = param["height"]
    height_values.append(height)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = height_label+f": {height}$L$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(height_values)), key=lambda i: height_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
height_values_sorted = [height_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]



output_file = os.path.join(script_path, '01_A_vs_t_all_varying_height.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $"+ velocity_bc_label,ylabel=label_crack_length,
                      usetex=True,
                      use_bw_palette=True,
                      show_markers=False,
                      x_range=[-0.1, 6.0],
                      y_range=[1.0,5.5])

output_file = os.path.join(script_path, "02_height_vs_initial_crack_length.png")
# ev.plot_multiple_lines(height_values_sorted,initial_crack_length_values_sorted,title="",x_label="height",y_label="initial crack length / $L$",output_file=output_file)
# output_file = os.path.join(script_path,"Jx_vs_wsteg_varying_stiffness.png")
# ev.plot_multiple_lines(height_values_sorted,initial_crack_length_values_sorted,x_label="$w_s$",y_label="Jx_max",output_file=output_file)




ev.plot_single_line(height_values_sorted,initial_crack_length_values_sorted,output_file,xlabel=height_label+" / $L$",ylabel=initial_crack_label+" / $L$")

data_file_name = "02.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(height_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)

data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
height_circular, init_crack_height_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
height_diamond, init_crack_height_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"02B_height_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([height_circular, height_diamond], 
                       [init_crack_height_circular, init_crack_height_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label=height_label+" / $L$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file)

simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"gc_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
gc_values = []
initial_crack_length_values = []



for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,2)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    gc = param["Gc_simulation"]
    gc_values.append(gc)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = f"$G_c$: {gc}"+"$G_c^{\text{ref}}$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(gc_values)), key=lambda i: gc_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
gc_values_sorted = [gc_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]

output_file = os.path.join(script_path, '03_A_vs_t_all_varying_gc.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length,
                      usetex=True,
                      use_bw_palette=True,
                      show_markers=False)

output_file = os.path.join(script_path, "04_gc_vs_initial_crack_length.png")
ev.plot_single_line(gc_values_sorted,initial_crack_length_values_sorted,output_file,xlabel="$G_c / G_c^{\text{ref}}$",ylabel=(initial_crack_label + " / $L$"),usetex=True,y_range=[0.0,0.8],x_range=[-0.1, 12.0])

data_file_name = "04.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(gc_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)

data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
data_circular, init_crack_length_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
data_diamond, init_crack_length_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"04B_gc_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([data_circular, data_diamond], 
                       [init_crack_length_circular, init_crack_length_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label="$G_c / G_c^{\text{ref}}$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file,
                       x_range=[0.0, 12.0])




simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"width_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
width_values = []
initial_crack_length_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,2)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    width = param["width"]
    width_values.append(width)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = width_label+f": {width}$L$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(width_values)), key=lambda i: width_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
width_values_sorted = [width_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]



output_file = os.path.join(script_path, '05A_A_vs_t_all_varying_width.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length, usetex=True,
                      use_bw_palette=True,
                      x_range=[-0.1,2.5],
                      y_range=[1.0, 3.5])


output_file = os.path.join(script_path, "05B_width_vs_initial_crack_length.png")
ev.plot_single_line(width_values_sorted,initial_crack_length_values_sorted,output_file,xlabel=(width_label+" / $L$"),ylabel=initial_crack_label+" / $L$",y_range=[0.0, 0.6], usetex=True)

data_file_name = "05.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(width_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)

data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
data_circular, init_crack_length_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
data_diamond, init_crack_length_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"05C_width_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([data_circular, data_diamond], 
                       [init_crack_length_circular, init_crack_length_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label=width_label+" / $L$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file)


simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"e_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
E_values = []
initial_crack_length_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
     # filter data so data row starts with initial brutal crack growth
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,2)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    la = param["lam_simulation"]
    mu = param["mue_simulation"]
    E = le.get_emod(la,mu)
    E_values.append(E)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = f"$E$: {E}$\mu$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(E_values)), key=lambda i: E_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
E_values_sorted = [E_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]



output_file = os.path.join(script_path, '06_A_vs_t_all_varying_E.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length,
                      x_range=[-0.1, 4.8])

output_file = os.path.join(script_path, "07_E_vs_initial_crack_length.png")
ev.plot_single_line(E_values_sorted,initial_crack_length_values_sorted,output_file,xlabel="$E / \mu^{\text{ref}}$",ylabel=initial_crack_label+" / $L$", y_range=[0.0, 0.6])

data_file_name = "07.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(E_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)

data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
data_circular, init_crack_length_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
data_diamond, init_crack_length_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"07B_E_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([data_circular, data_diamond], 
                       [init_crack_length_circular, init_crack_length_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label="$E / \mu^{\text{ref}}$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file,
                       y_range=[0.0, 0.6])



simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"eps_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
eps_values = []
initial_crack_length_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,5)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    
    eps = param["eps"]

    eps_values.append(eps)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = f"$eps$: {eps}$L$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(eps_values)), key=lambda i: eps_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
eps_values_sorted = [eps_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]



output_file = os.path.join(script_path, '08_A_vs_t_all_varying_eps.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length,
                      x_range=[-0.1, 4.0],
                      y_range=[1.0,3.5])

output_file = os.path.join(script_path, "09_eps_vs_initial_crack_length.png")
ev.plot_single_line(eps_values_sorted,initial_crack_length_values_sorted,output_file,xlabel="eps / $L$",ylabel=initial_crack_label+" / $L$", y_range=[0.0, 0.6])


data_file_name = "09.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(eps_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)

data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
data_circular, init_crack_length_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
data_diamond, init_crack_length_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"09B_eps_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([data_circular, data_diamond], 
                       [init_crack_length_circular, init_crack_length_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label="$\epsilon$ / $L$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file)
# def scatter_plot_with_classes(
#     x_values, y_values, 
#     marker_values, color_values, text_values, 
#     marker_title='Marker', color_title='Color', 
#     marker_fontsize=10, color_fontsize=10, text_fontsize=8, 
#     output_file='scatter_plot.png'
# ):
#     # Convert all inputs to numpy arrays if they aren't already
#     x_values = np.asarray(x_values)
#     y_values = np.asarray(y_values)
#     marker_values = np.asarray(marker_values)
#     color_values = np.asarray(color_values)
#     text_values = np.asarray(text_values)

#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     unique_marker_classes = np.unique(marker_values)
#     unique_color_classes = np.unique(color_values)
    
#     markers = ['o', 's', '^', 'D', 'X', 'P', 'v', '<', '>', 'h']
#     marker_dict = {v: markers[i % len(markers)] for i, v in enumerate(unique_marker_classes)}
    
#     color_palette = sns.color_palette("hsv", len(unique_color_classes))
#     color_dict = {v: color_palette[i] for i, v in enumerate(unique_color_classes)}
    
#     for marker_class in unique_marker_classes:
#         for color_class in unique_color_classes:
#             indices = (marker_values == marker_class) & (color_values == color_class)
#             if np.any(indices):  # Only plot if there are matching indices
#                 ax.scatter(
#                     x_values[indices], y_values[indices], 
#                     marker=marker_dict[marker_class], color=color_dict[color_class], 
#                     label=f"{marker_title} {marker_class}, {color_title} {color_class}"
#                 )
    
#     for i, txt in enumerate(text_values):
#         ax.text(
#             x_values[i], y_values[i], str(txt),
#             fontsize=text_fontsize, ha='right', va='bottom'
#         )
    
#     marker_legend = [Line2D([0], [0], marker=marker_dict[m], color='w', markerfacecolor='k', markersize=10, label=f'{marker_title} {m}') 
#                      for m in unique_marker_classes]
    
#     color_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[c], markersize=10, label=f'{color_title} {c}') 
#                     for c in unique_color_classes]
    
#     ax.legend(handles=marker_legend + color_legend, fontsize=marker_fontsize + 2, loc='upper right')
    
#     ax.set_xlabel('X Axis', fontsize=marker_fontsize)
#     ax.set_ylabel('Y Axis', fontsize=marker_fontsize)
#     ax.set_title('Scatter Plot with Marker and Color Classification', fontsize=marker_fontsize + 4)
    
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()
    
    
    
# simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"sigc_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# # output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
# data_to_plot = []
# legend_entries = []
# eps_values = []
# gc_values = []
# E_values = []
# sigc_values = []
# initial_crack_length_values = []

# for sim in simulation_results:
#     data = sim[0]
#     param = sim[1]
#     data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
#     initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
#     initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
#     initial_crack_length_values.append(initial_crack_length)
    
#     eps = param["eps"]
#     eps_values.append(eps)
    
#     gc = param["Gc_simulation"]
#     gc_values.append(gc)
    
#     la = param["lam_simulation"]
#     mu = param["mue_simulation"]
#     E = le.get_emod(la,mu)
#     E_values.append(E)
    
#     sigc = pf.sig_c_quadr_deg(gc,mu,eps)
#     sigc_values.append(sigc)
    
#     data_to_plot.append(data)
#     legend_entry = f"$eps$: {eps} $gc$: {gc}; $E$: {E}; $sigc$: {sigc};"
#     legend_entries.append(legend_entry)
    
# sorted_indices = sorted(range(len(eps_values)), key=lambda i: eps_values[i])
# data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
# legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
# eps_values_sorted = [eps_values[i] for i in sorted_indices]
# gc_values_sorted = [gc_values[i] for i in sorted_indices]
# E_values_sorted = [E_values[i] for i in sorted_indices]
# sigc_values_sorted = [sigc_values[i] for i in sorted_indices]
# initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]



# output_file = os.path.join(script_path, '10_A_vs_t_all_varying_sigc.png')  
# ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
#                       col_x=0,
#                       col_y=9,
#                       output_filename=output_file,
#                       legend_labels=legend_entries_sorted,
#                       xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length)


# output_file = os.path.join(script_path, '11_cracklength_vs_sigc.png') 
# scatter_plot_with_classes(sigc_values_sorted,
#                           initial_crack_length_values_sorted,
#                           eps_values_sorted,
#                           gc_values_sorted,
#                           E_values_sorted,
#                           "eps",
#                           "gc",output_file=output_file)


simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"nu_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
nu_values = []  # Collect Poisson ratios
initial_crack_length_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
     # filter data so data row starts with initial brutal crack growth
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,5)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    la = param["lam_simulation"]
    mu = param["mue_simulation"]
    nu = le.get_nu(la, mu)  # Calculate Poisson ratio
    nu_values.append(nu)
    E = le.get_emod(la,mu)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = f"$\\nu$: {nu:.2f}, $E$: {E:.1f}$\mu$"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(nu_values)), key=lambda i: nu_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
nu_values_sorted = [nu_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]

# Plot A vs t for varying \nu
output_file = os.path.join(script_path, '12_A_vs_t_all_varying_nu.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length,
                      y_range=[1.0, 3.5],
                      x_range=[-0.1, 4.0])

# Plot \nu vs initial crack length
output_file = os.path.join(script_path, "13_nu_vs_initial_crack_length.png")
ev.plot_single_line(nu_values_sorted,initial_crack_length_values_sorted,output_file,xlabel="$\\nu$",ylabel=initial_crack_label+" / $L$", y_range=[0.0, 0.6])

data_file_name = "13.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(nu_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)
data_file_circular = os.path.join(script_path,"..", circular_holes_folder_name ,data_file_name)
data_circular, init_crack_length_circular = ev.load_plot_data_from_file(data_file_circular)
data_file_diamond = os.path.join(script_path,"..", diamond_folder_name ,data_file_name)
data_diamond, init_crack_length_diamond = ev.load_plot_data_from_file(data_file_diamond)
output_file = os.path.join(script_path,"13B_nu_vs_initial_crack_length_circluar&diamond.png")
ev.plot_multiple_lines([data_circular, data_diamond], 
                       [init_crack_length_circular, init_crack_length_diamond],
                       legend_labels=[circular_label, diamond_label],
                       x_label="$\\nu$",y_label=initial_crack_label+" / $L$",
                       output_file=output_file)


simulation_results = ev.read_all_simulation_data(os.path.join(script_path,"e_gc_study"),graphs_filename='run_simulation_K_sym_graphs.txt')

# output_file = os.path.join(script_path, 'Jx_vs_xct_all.png')
data_to_plot = []
legend_entries = []
E_values = []
Gc_values = []
initial_crack_length_values = []

for sim in simulation_results:
    data = sim[0]
    param = sim[1]
    
    # filter data so data row starts with initial brutal crack growth
    # data, A_initial_num, t_start_brutal_crack_growth, Delta_t_initial_crack_growth = filter_data_to_range_where_A_exceeds_initial_value(filter_data, init_crack, data, param)
    
    # initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=9, x_threshold=t_start_brutal_crack_growth+Delta_t_initial_crack_growth)[1]
    # initial_crack_length = initial_crack_length - A_initial_num # include effect of initial crack
    crack_length_until_hole = param["width"]/2.0 - DHOLE/2.0
    data = filter_data(data,3,0.0,0.95*param["width"])
    data = remove_first_percentage(data,5)
    data_crack_length = filter_data(data,3,crack_length_until_hole+0.1*DHOLE,param["width"])
    initial_crack_length = ev.get_initial_crack_length(find_max_y_under_x_threshold, data_crack_length,y_col=col_crack_length)
    
    la = param["lam_simulation"]
    mu = param["mue_simulation"]
    E = le.get_emod(la,mu)
    Gc = param["Gc_simulation"]
    E_values.append(E)
    Gc_values.append(Gc)
    initial_crack_length_values.append(initial_crack_length)
    data_to_plot.append(data)
    legend_entry = f"$(E, G_c)$: {E:.1f}$\mu$, {Gc:.1f}"+"$G_c^{\text{ref}}$)"
    legend_entries.append(legend_entry)
    
sorted_indices = sorted(range(len(E_values)), key=lambda i: E_values[i])
data_to_plot_sorted = [data_to_plot[i] for i in sorted_indices]
legend_entries_sorted = [legend_entries[i] for i in sorted_indices]
E_values_sorted = [E_values[i] for i in sorted_indices]
Gc_values_sorted = [Gc_values[i] for i in sorted_indices]
E_Gc_values_sorted = [E_values[i]*Gc_values[i] for i in sorted_indices]
initial_crack_length_values_sorted = [initial_crack_length_values[i] for i in sorted_indices]


output_file = os.path.join(script_path, '14_A_vs_t_all_varying_E_Gc.png')  
ev.plot_multiple_columns(data_objects=data_to_plot_sorted,
                      col_x=0,
                      col_y=col_crack_length,
                      output_filename=output_file,
                      legend_labels=legend_entries_sorted,
                      xlabel="$t / ( L / $" + velocity_bc_label,ylabel=label_crack_length,
                      y_range=[1.0, 3.5],
                      x_range=[-0.1, 4.0])

output_file = os.path.join(script_path, "15_initial_crack_length_vs_ExGc.png")
ev.plot_single_line(E_Gc_values_sorted,initial_crack_length_values_sorted,output_file,xlabel="$EG_c / (\mu{G_c^{\text{ref}}})$",ylabel=initial_crack_label+" / $L$",plot_type="dot",y_range=[0.0, 0.6])

data_file_name = "15.data"
output_file = os.path.join(script_path, data_file_name)
ev.save_plot_data_to_file(E_Gc_values_sorted,y=initial_crack_length_values_sorted,filename=output_file)




