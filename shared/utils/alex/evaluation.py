import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
import os
import math
import numpy as np
import re
from collections import defaultdict
from typing import Callable, List, Dict, Tuple

import pandas as pd
import alex.postprocessing as pp


### AUXILIARY 
def compute_gc_num(mesh_name, gc, eps_factor, h_all, reference_L_global):
    """
    Computes the gc_num value.

    :param mesh_name: Name of the mesh type.
    :param gc: Gc value.
    :param eps: Epsilon value.
    :param h_all: Dictionary mapping mesh types to their corresponding h values.
    :return: Computed gc_num value.
    """
    h_value = h_all[mesh_name]
    eps =  get_eps(reference_L_global, eps_factor)
    return gc * (1.0 + h_value / ( eps * 4.0))

def get_gc_num_for_key(key, h_all, reference_L_global):
    params = extract_parameters(key)
    if params:
        mesh_type = params[0]
        lam_value = params[1]
        mue_value = params[2]
        gc_value = params[3]
        eps_value = params[4]
        gc_num_value = compute_gc_num(mesh_name=mesh_type, gc=gc_value, eps_factor=eps_value, h_all=h_all, reference_L_global=reference_L_global)
        return gc_num_value
        # return gc_value

def get_eps(reference_L_global, eps_factor):
    eps = reference_L_global / eps_factor
    return eps

def get_pore_size_eps_ratio(mesh_name, eps_value, pore_size_all):
    pore_size = pore_size_all[mesh_name]
    ratio = pore_size / eps_value
    return ratio

def sig_c_quadr_deg(Gc, mu, epsilon):
        return 9.0/16.0 * math.sqrt(Gc*2.0*mu/(6.0*epsilon))

def get_sig_c(extract_parameters,keys, reference_L_global):

    
    def get_epsilon(eps_factor):
        refernce_L = reference_L_global 
        return refernce_L/eps_factor
    
    sig_c = np.zeros_like(keys,dtype=float)
    
    for i in range(0,len(keys)):
        key = keys[i]
        mue = extract_parameters(key)[2]
        gc = extract_parameters(key)[3]
        eps_factor = extract_parameters(key)[4]
        sig_c[i] =  sig_c_quadr_deg(mu=mue,Gc=gc,epsilon=get_epsilon(eps_factor))
    return sig_c

def poisson_ratio(lam, mue):
    nu = lam / (2 * (lam + mue))
    return nu




### FILTERING DATA 

def create_results_dict(directory_path, outer_pattern):
    results_dict = {}
    first_level_keys = set()


# Iterate over all the directories
    for folder_name in os.listdir(directory_path):
        outer_match = outer_pattern.match(folder_name)
        if outer_match:
        # Extract the values from the outer directory name
            mesh_name = outer_match.group("mesh_name")
            lam_value = float(outer_match.group("lam_value"))
            mue_value = float(outer_match.group("mue_value"))
            Gc_value = float(outer_match.group("Gc_value"))
            eps_value = float(outer_match.group("eps_value"))
            order_value = int(outer_match.group("order_value"))

        # Get the path of the outer folder
            outer_folder_path = os.path.join(directory_path, folder_name)

        # Initialize variables
            data_file_path = None
            first_level_found = False

        # Check for any file ending with *_graphs.txt at the first level
            for item in os.listdir(outer_folder_path):
                if item.endswith("_graphs.txt") and os.path.isfile(os.path.join(outer_folder_path, item)):
                    data_file_path = os.path.join(outer_folder_path, item)
                    first_level_found = True
                    break

        # If not found at the first level, check within the inner folder
            if not first_level_found:
                inner_folder_name = None
                for item in os.listdir(outer_folder_path):
                    if item.startswith("simulation_") and os.path.isdir(os.path.join(outer_folder_path, item)):
                        inner_folder_name = item
                        break

                if inner_folder_name:
                    inner_folder_path = os.path.join(outer_folder_path, inner_folder_name)
                    for item in os.listdir(inner_folder_path):
                        if item.endswith("_graphs.txt") and os.path.isfile(os.path.join(inner_folder_path, item)):
                            data_file_path = os.path.join(inner_folder_path, item)
                            break

        # If a data file was found, process it
            if data_file_path and os.path.isfile(data_file_path):
            # Read the data from the file
                data = []
                with open(data_file_path, 'r') as file:
                    for line in file:
                    # Skip lines that start with #
                        if line.startswith('#'):
                            continue
                    # Split the line into columns and convert to float
                        data.append(list(map(float, line.split())))

            # Store the data in the dictionary
                key = f"{mesh_name}_lam{lam_value}_mue{mue_value}_Gc{Gc_value}_eps{eps_value}_order{order_value}"
                results_dict[key] = np.array(data)

            # If the file was found at the first level, add the key to first_level_keys
                if first_level_found:
                    first_level_keys.add(key)
    return results_dict,first_level_keys


def create_max_dict(results_dict, column_index=1):
    """
    Creates a new dictionary that stores the maximum value of a specified column for each key in the results dictionary.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param column_index: Index of the column for which to find the maximum value. Default is 1.
    :return: Dictionary with the same keys as results_dict and maximum values of the specified column as values.
    """
    max_dict = {}
    
    for key, data in results_dict.items():
        max_value = np.max(data[:, column_index])
        max_dict[key] = max_value
    
    return max_dict

# Your function to extract parameters
def extract_parameters(key: str) -> Tuple[str, float, float, float, float, int]:

    # Regular expression pattern to match the key format
    pattern = re.compile(
        r"(?P<mesh_name>[a-zA-Z0-9_]+)_lam(?P<lam_value>\d+\.\d+)_mue(?P<mue_value>\d+\.\d+)_Gc(?P<Gc_value>\d+\.\d+)_eps(?P<eps_value>\d+\.\d+)_order(?P<order_value>\d+)"
    )
    
    match = pattern.match(key)
    if match:
        # Extract the values and convert them to appropriate types
        mesh_name = match.group("mesh_name")
        lam_value = float(match.group("lam_value"))
        mue_value = float(match.group("mue_value"))
        Gc_value = float(match.group("Gc_value"))
        eps_value = float(match.group("eps_value"))
        order_value = int(match.group("order_value"))
        return (mesh_name, lam_value, mue_value, Gc_value, eps_value, order_value)
    else:
        raise ValueError("Key format is incorrect")
    
def get_mesh_name(key: str):
    return extract_parameters(key)[0]

def get_lam(key: str):
    return extract_parameters(key)[1]

def get_mue(key: str):
    return extract_parameters(key)[2]

def get_gc(key: str):
    return extract_parameters(key)[3]

def get_order(key: str):
    return extract_parameters(key)[4]
    
def group_by_function(keys: List[str], grouping_function: Callable[[Tuple[str, float, float, float, float, int]], any]) -> Dict[any, List[str]]:
    groups = defaultdict(list)
    
    for key in keys:
        try:
            parameters = extract_parameters(key)
            group_key = grouping_function(parameters)
            groups[group_key].append(key)
        except ValueError as e:
            print(f"Skipping key {key}: {e}")
    
    return dict(groups)
    
def filter_keys(results_dict, target_Gc=None, target_eps=None, target_lam=None, target_mue=None, target_mesh_types=None):
    filtered_keys = []
    for key in results_dict.keys():
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            Gc_value = params[3]
            eps_value = params[4]
            lam_value = params[1]
            mue_value = params[2]
            
            # Check if the key meets all specified criteria
            if (target_Gc is None or np.isclose(Gc_value, target_Gc).any()) and \
               (target_eps is None or np.isclose(eps_value, target_eps).any()) and \
               (target_lam is None or np.isclose(lam_value, target_lam).any()) and \
               (target_mue is None or np.isclose(mue_value, target_mue).any()) and \
               (target_mesh_types is None or mesh_type in target_mesh_types):
                filtered_keys.append(key)
    return filtered_keys

def remove_parameter(param_string, param_names):
    """
    Removes the specified parameters and their values from the parameter string.

    :param param_string: The parameter string in the format 
                         "mesh_name_lam{lam_value}_mue{mue_value}_Gc{Gc_value}_eps{eps_value}_order{order_value}"
    :param param_names: List of lists/tuples of parameter names to remove (e.g., [['mesh_name'], ['lam', 'mue'], ['Gc']], etc.)
    :return: The parameter string with the specified parameters removed.
    """
    # Split the parameter string into parts
    parts = param_string.split('_')
    
    # Initialize indices to remove
    indices_to_remove = []
    
    # Handle removal of 'mesh_name' separately since it's at the beginning
    if ['mesh_name'] in param_names:
        indices_to_remove.append(0)  # Add index 0 for 'mesh_name'
    
    # Identify the indices of other parameters to remove
    for param_set in param_names:
        if param_set == ['mesh_name']:
            continue  # Skip since we've already handled 'mesh_name'
        
        found = False
        for i, part in enumerate(parts):
            if any(part.startswith(param_name) for param_name in param_set):
                found = True
                indices_to_remove.append(i)
                if len(part) > len(max(param_set, key=len)):  # Check length of longest parameter name in set
                    # Also remove the value if it exists after the parameter name
                    indices_to_remove.append(i + 1)
                break
        if not found:
            raise ValueError(f"No parameters '{param_set}' found in the parameter string")
    
    # Remove identified parameters and their values
    new_parts = [part for i, part in enumerate(parts) if i not in indices_to_remove]
    
    # Reconstruct the parameter string without the specified parameters
    new_param_string = '_'.join(new_parts)
    
    return new_param_string



### PLOTTING
def plot_results(results_dict, keys, column_index, save_path, scaling_factors=None):
    """
    Plots a given result column vs time for a set of keys and saves the plot to disk.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param column_index: Index of the result column to plot against time.
    :param save_path: File path to save the plot.
    :param scaling_factors: List of scaling factors for each key. If None, no scaling is applied.
    """
    if scaling_factors is None:
        scaling_factors = [1.0] * len(keys)
    
    if len(keys) != len(scaling_factors):
        raise ValueError("The length of keys and scaling_factors must be the same.")
    
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better readability

    for key, scaling_factor in zip(keys, scaling_factors):
        if key in results_dict:
            data = results_dict[key]
            time = data[:, 0]
            result_column = data[:, column_index] * scaling_factor  # Apply scaling
            
            plt.plot(time, result_column, label=f"{key} (scaled by {scaling_factor})")
        else:
            print(f"Key '{key}' not found in the results dictionary.")

    plt.xlabel('Time')
    plt.ylabel(f'Column {column_index}')
    # plt.title(f'Results for Column {column_index} vs Time')
    plt.legend()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot to disk
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory
    
# Example function modified for your specific plotting needs
def plot_max_Jx_vs_sig_c(results_dict, keys, plot_title, save_path, reference_L_global, pore_size_all, special_keys=None, h_all = None):
    # Create the max dictionary
    max_Jx_dict = create_max_dict(results_dict, column_index=1)
    
    # Compute sig_c for all keys in max_Jx_dict
    sig_c_values = get_sig_c(extract_parameters, keys, reference_L_global=reference_L_global)
    
    # Create a colormap for the mesh types
    mesh_colors = {
        "fine_pores": mcolors.to_rgba('darkred'),
        "medium_pores": mcolors.to_rgba('red'),
        "coarse_pores": mcolors.to_rgba('lightcoral')
    }
    
    # Marker types based on (lam, mue) values
    marker_types = {
        (1.0, 1.0): 'o',       # Circle
        (1.5, 1.0): 'p',      # Cross
        (0.6667, 1.0): '^',      # Cross
        (1.0, 1.5): 'v',      # Dot
        (10.0, 10.0): '^',     # Triangle
        (15.0, 10.0): 's'      # Square
    }
    
    # Plot max_Jx_dict values vs sig_c_values with colors and markers based on mesh type and (lam, mue)
    plt.figure(figsize=(10, 20))
    
    unique_labels = set()
    
    for key, sig_c in zip(keys, sig_c_values):
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            lam_mue = (params[1], params[2])
            
            color = mesh_colors.get(mesh_type, 'black')
            marker = marker_types.get(lam_mue, 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
            label = mesh_type if mesh_type not in unique_labels else ""
            
            # Check if the current key is in the special_keys set
            if special_keys and key in special_keys:
                edge_color = 'black'
                linewidth = 2.0
            else:
                edge_color = 'none'
                linewidth = 0
            
            y_value = max_Jx_dict[key] / get_gc_num_for_key(key,h_all=h_all,reference_L_global=reference_L_global)
            plt.scatter(sig_c, y_value,
                        color=color, marker=marker, s=100, label=label, edgecolor=edge_color, linewidth=linewidth)
            unique_labels.add(mesh_type)
            
            # Display Gc and inverse of eps values as tuples to the right of the markers
            eps_value = get_eps(reference_L_global,params[4])
            eps_pore_size_ratio = get_pore_size_eps_ratio(mesh_type,eps_value, pore_size_all)
            plt.text(sig_c, y_value, f"({params[3]}, {eps_pore_size_ratio})", fontsize=9, ha='left', va='bottom')
    
    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles
    for mesh_type, color in mesh_colors.items():
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=mesh_type))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        nu = poisson_ratio(lam, mue)
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue} (nu={nu:.2f})'))
    
    plt.xlabel('Sig_c')
    plt.ylabel('Max Jx / Gc_num')
    plt.title(plot_title)
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    

def plot_max_Jx_vs_data(results_dict, keys, xdata_function, xdata_label, data_label_function,plot_title, save_path, special_keys=None, reference_L_global = None, h_all=None):
    # Create the max dictionary
    max_Jx_dict = create_max_dict(results_dict, column_index=1)
    
    # Initialize the eps_pore_size_ratio values for each key
    xdata_values = dict()

    # Compute eps_pore_size_ratio for all keys in max_Jx_dict
    for key in keys:
        xdata_values[key] = xdata_function(key)
            
    
    # Create a colormap for the mesh types
    mesh_colors = {
        "fine_pores": mcolors.to_rgba('darkred'),
        "medium_pores": mcolors.to_rgba('red'),
        "coarse_pores": mcolors.to_rgba('lightcoral')
    }
    
    # Marker types based on (lam, mue) values
    marker_types = {
        (1.0, 1.0): 'o',      # Circle
        (1.5, 1.0): 'p',      # Cross
        (1.0, 1.5): 'v',      # Dot
        (10.0, 10.0): '^',    # Triangle
        (15.0, 10.0): 's'     # Square
    }
    
    # Plot max_Jx_dict values vs eps_pore_size_ratio_values with colors and markers based on mesh type and (lam, mue)
    plt.figure(figsize=(10, 20))
    
    unique_labels = set()
    
    for key in keys:
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            lam_mue = (params[1], params[2])
            
            color = mesh_colors.get(mesh_type, 'black')
            marker = marker_types.get(lam_mue, 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
            label = mesh_type if mesh_type not in unique_labels else ""
            
            # Check if the current key is in the special_keys set
            if special_keys and key in special_keys:
                edge_color = 'black'
                linewidth = 2.0
            else:
                edge_color = 'none'
                linewidth = 0
            
            y_value = max_Jx_dict[key] / get_gc_num_for_key(key,h_all=h_all,reference_L_global=reference_L_global)
            plt.scatter(xdata_values[key], y_value, color=color, marker=marker, s=100, label=label, edgecolor=edge_color, linewidth=linewidth)
            unique_labels.add(mesh_type)
            
            # Display Gc and inverse of eps values as tuples to the right of the markers
            data_label1, data_label2 = data_label_function(key)
            plt.text(xdata_values[key], y_value, f"({data_label1}, {data_label2})", fontsize=9, ha='left', va='bottom')
    
    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles
    for mesh_type, color in mesh_colors.items():
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=mesh_type))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        nu = poisson_ratio(lam, mue)
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue} (nu={nu:.2f})'))
    
    plt.xlabel(xdata_label)
    plt.ylabel('Max Jx / Gc_num')
    plt.title(plot_title)
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()    

    
def plot_max_Jx_vs_pore_size_eps_ratio_multiple(
    results_dicts, 
    keys_list, 
    plot_title, 
    save_path, 
    pore_size_all, 
    reference_L_global, 
    special_keys_list=None, 
    h_all_list=None, 
    axis_label_size=18,     # Parameter for axis label size
    tick_label_size=16,     # Parameter for tick label size
    legend_label_size=16,   # Parameter for legend label size
    number_size=23,         # Parameter for size of numbers (e.g., on the markers)
    marker_size=200,        # Parameter for marker size
    fig_size=(10, 20),      # Parameter for figure size
    show_numbers=True,      # Parameter to toggle number display
    x_label="pore_size_eps_ratio",   # New parameter for x-axis label
    y_label="Max Jx / Gc_num"        # New parameter for y-axis label
):
    # Define color palettes for different datasets
    color_palettes = [
        {"fine_pores": mcolors.to_rgba('darkred'), "medium_pores": mcolors.to_rgba('red'), "coarse_pores": mcolors.to_rgba('lightcoral')},
        {"fine_pores": mcolors.to_rgba('darkblue'), "medium_pores": mcolors.to_rgba('blue'), "coarse_pores": mcolors.to_rgba('lightblue')},
        {"fine_pores": mcolors.to_rgba('darkgreen'), "medium_pores": mcolors.to_rgba('green'), "coarse_pores": mcolors.to_rgba('lightgreen')},
        {"fine_pores": mcolors.to_rgba('darkorange'), "medium_pores": mcolors.to_rgba('orange'), "coarse_pores": mcolors.to_rgba('lightsalmon')},
    ]

    plt.figure(figsize=fig_size)
    unique_labels = set()
    used_marker_styles = set()
    used_pore_types = set()

    # Iterate over each results_dict and corresponding keys
    for i, (results_dict, keys) in enumerate(zip(results_dicts, keys_list)):
        max_Jx_dict = create_max_dict(results_dict, column_index=1)
        pore_size_eps_ratio_values = []

        for key in keys:
            params = extract_parameters(key)
            if params:
                mesh_type = params[0]
                eps_value = get_eps(reference_L_global, params[4])
                pore_size_eps_ratio = get_pore_size_eps_ratio(mesh_type, eps_value, pore_size_all)
                pore_size_eps_ratio_values.append(pore_size_eps_ratio)

        mesh_colors = color_palettes[i % len(color_palettes)]
        marker_types = {
            (1.0, 1.0): 'o', (1.5, 1.0): 'p', (0.6667, 1.0): 'v', (10.0, 10.0): '^', (15.0, 10.0): 's'
        }

        for key, pore_size_eps_ratio in zip(keys, pore_size_eps_ratio_values):
            params = extract_parameters(key)
            if params:
                mesh_type = params[0]
                lam_mue = (params[1], params[2])
                color = mesh_colors.get(mesh_type, 'black')
                marker = marker_types.get(lam_mue, 'x')
                label = f"{mesh_type} (Set {i+1})" if mesh_type not in unique_labels else ""
                
                if special_keys_list and key in special_keys_list[i]:
                    edge_color = 'black'
                    linewidth = 2.0
                else:
                    edge_color = 'none'
                    linewidth = 0

                y_value = max_Jx_dict[key] / get_gc_num_for_key(key, h_all=h_all_list[i], reference_L_global=reference_L_global)
                plt.scatter(
                    pore_size_eps_ratio, y_value, 
                    color=color, marker=marker, s=marker_size,
                    label=label, edgecolor=edge_color, linewidth=linewidth
                )
                unique_labels.add(mesh_type)
                used_marker_styles.add((lam_mue, marker))
                used_pore_types.add(mesh_type)

                if show_numbers:
                    plt.text(pore_size_eps_ratio, y_value, f"({params[3]}, {pore_size_eps_ratio})", fontsize=number_size, ha='left', va='bottom')

    handles = []
    
    for i, _ in enumerate(results_dicts):
        mesh_colors = color_palettes[i % len(color_palettes)]
        for mesh_type, color in mesh_colors.items():
            if mesh_type in used_pore_types:
                handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=f'{mesh_type} (Set {i+1})'))
    
    for (lam, mue), marker in marker_types.items():
        if (lam, mue) in [m for m, _ in used_marker_styles]:
            nu = poisson_ratio(lam, mue)
            handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue} (nu={nu:.2f})'))

    # Use new x_label and y_label parameters
    plt.xlabel(x_label, fontsize=axis_label_size)
    plt.ylabel(y_label, fontsize=axis_label_size)
    plt.title(plot_title, fontsize=axis_label_size)

    plt.xticks(fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)
    plt.legend(handles=handles, loc='best', fontsize=legend_label_size)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()





    


def plot_results_2yaxis(results_dict, keys, column_indices, save_path, scaling_factors=None, y_labels=None, show_legend=True, label_fontsize=16, tick_fontsize=12):
    """
    Plots up to two result columns vs time for a set of keys and saves the plot to disk.
    The first column is plotted on the left y-axis and the second on the right y-axis.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param column_indices: List or tuple of up to two column indices to plot against time.
    :param save_path: File path to save the plot.
    :param scaling_factors: List of scaling factors for each key. If None, no scaling is applied.
    :param y_labels: List or tuple of y-axis labels. The first label is for the left y-axis, 
                     the second (if any) is for the right y-axis. If None, defaults are used.
    :param show_legend: Boolean to determine whether to show the legend. Default is True.
    :param label_fontsize: Font size for the axis labels. Default is 12.
    :param tick_fontsize: Font size for the axis tick numbers. Default is 10.
    """
    if scaling_factors is None:
        scaling_factors = [1.0] * len(keys)
    
    if len(keys) != len(scaling_factors):
        raise ValueError("The length of keys and scaling_factors must be the same.")
    
    if len(column_indices) > 2:
        raise ValueError("column_indices must contain one or two indices.")

    if y_labels is None:
        y_labels = [f'Column {col_index}' for col_index in column_indices]
    elif len(y_labels) != len(column_indices):
        raise ValueError("The length of y_labels must match the length of column_indices.")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Set the figure size for better readability

    ax2 = None
    if len(column_indices) > 1:
        ax2 = ax1.twinx()  # Create a secondary y-axis

    for key, scaling_factor in zip(keys, scaling_factors):
        if key in results_dict:
            data = results_dict[key]
            time = data[:, 0]

            # Plot the first column on the left y-axis
            result_column1 = data[:, column_indices[0]] * scaling_factor
            ax1.plot(time, result_column1, label=f"{key} (scaled by {scaling_factor})")

            if len(column_indices) > 1:
                # Plot the second column on the right y-axis
                result_column2 = data[:, column_indices[1]] * scaling_factor
                ax2.plot(time, result_column2, linestyle='--', label=f"{key} (scaled by {scaling_factor}, secondary)")
                
        else:
            print(f"Key '{key}' not found in the results dictionary.")

    ax1.set_xlabel('Time', fontsize=label_fontsize)
    ax1.set_ylabel(y_labels[0], fontsize=label_fontsize)  # Left y-axis label
    ax1.tick_params(axis='both', labelsize=tick_fontsize)  # Set font size for ticks on left y-axis

    if len(column_indices) > 1 and ax2 is not None:
        ax2.set_ylabel(y_labels[1], fontsize=label_fontsize)  # Right y-axis label
        ax2.tick_params(axis='both', labelsize=tick_fontsize)  # Set font size for ticks on right y-axis

    # Combine legends from both axes if show_legend is True
    if show_legend:
        if ax2 is not None:
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot to disk
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory
    
def plot_gc_num_vs_gc(keys, h_all, save_path, reference_L_global, mesh_colors):
    """
    Plots gc_num vs Gc for a set of keys and saves the plot to disk.

    :param results_dict: Dictionary containing the results data.
                         The keys are folder names and values are numpy arrays with the data.
    :param keys: List of keys to plot.
    :param h_all: Dictionary mapping mesh types to their corresponding h values.
    :param save_path: File path to save the plot.
    """
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size for better readability

    # mesh_colors = {
    #     "fine_pores": mcolors.to_rgba('darkred'),
    #     "medium_pores": mcolors.to_rgba('red'),
    #     "coarse_pores": mcolors.to_rgba('lightcoral')
    # }
    
    marker_types = {
        (1.0, 1.0): 'o',       # Circle
        (1.5, 1.0): 'p',      # Cross
        (1.0, 1.5): 'v',      # Dot
        (10.0, 10.0): '^',     # Triangle
        (15.0, 10.0): 's'      # Square
    }
    
    unique_labels = set()
    
    for key in keys:
        params = extract_parameters(key)
        if params:
            mesh_type = params[0]
            lam_value = params[1]
            mue_value = params[2]
            gc_value = params[3]
            eps_value = params[4]
            
            gc_num_value = compute_gc_num(mesh_name=mesh_type, gc=gc_value, eps_factor=eps_value, h_all=h_all, reference_L_global=reference_L_global)
            
            color = mesh_colors.get(mesh_type, 'black')
            marker = marker_types.get((lam_value, mue_value), 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
            label = mesh_type if mesh_type not in unique_labels else ""
            
            plt.scatter(gc_value, gc_num_value, color=color, marker=marker, s=100, label=label)
            unique_labels.add(mesh_type)
            
            # Display gc_num values as text next to the markers
            plt.text(gc_value, gc_num_value, f"{1.0/eps_value:.2f}", fontsize=9, ha='right', va='bottom')
    
    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles
    for mesh_type, color in mesh_colors.items():
        handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=mesh_type))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue}'))
    
    plt.xlabel('Gc')
    plt.ylabel('gc_num')
    plt.title('gc_num vs Gc')
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    
def plot_eps_h_ratio(keys, h_all, reference_L_global, mesh_colors, output_file, lower_limit_eps_h=0.0, lower_limit_pore_size_eps=0.0, pore_size_all=None):
    ratios = []
    epss = []
    colors = []
    outlines = []
    
    for key in keys:
        params = extract_parameters(key)
        eps_factor = float(params[4])
        mesh_type = params[0]
        if eps_factor is not None and mesh_type in h_all:
            h = h_all[mesh_type]
            eps = get_eps(reference_L_global, eps_factor)
            ratio = eps / h
            ratios.append(ratio)
            epss.append(eps)
            colors.append(mesh_colors[mesh_type])
            
            # Determine if the point meets the criteria
            pore_size = pore_size_all[mesh_type]
            ratio_pore_size_eps = pore_size / eps
            if ratio >= lower_limit_eps_h and ratio_pore_size_eps >= lower_limit_pore_size_eps:
                outlines.append('green')
            else:
                outlines.append('none')
    
    plt.figure(figsize=(10, 6))
    
    for eps, ratio, color, outline in zip(epss, ratios, colors, outlines):
        plt.scatter(eps, ratio, c=[color], edgecolors=outline, marker='o', linewidths=1.5)
    
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\epsilon/h$')
    plt.title(r'Ratio of $\epsilon/h$ vs $\epsilon$')
    
    # Create custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mesh_colors[mesh], markersize=10, label=mesh) for mesh in mesh_colors]
    plt.legend(handles=handles, title="Mesh Types")
    
    plt.grid(True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.close()
    
    
def plot_ratio_pore_size_eps(results_dict, save_path, reference_L_global, pore_size_all, mesh_colors):
    plt.figure(figsize=(10, 6))
    
    for key in results_dict.keys():
        params = extract_parameters(key)
        mesh_name = params[0]
        eps_factor = float(params[4])
        eps_value = get_eps(reference_L_global,eps_factor=eps_factor)
        ratio = get_pore_size_eps_ratio(mesh_name, eps_value, pore_size_all=pore_size_all)
        plt.scatter(eps_value, ratio, color=mesh_colors[mesh_name], label=mesh_name)
        
    plt.xlabel('eps')
    plt.ylabel('pore_size/eps')
    plt.title('Ratio of Pore Size to Eps for All Mesh Types')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



def plot_ratio_pore_size_h(results_dict, save_path, h_all, pore_size_all, mesh_colors):
    plt.figure(figsize=(10, 6))
    
    for key in results_dict.keys():
        params = extract_parameters(key)
        mesh_name = params[0]
        h_value = h_all[mesh_name]
        pore_size = pore_size_all[mesh_name]
        ratio = pore_size / h_value
        plt.scatter(h_value, ratio, color=mesh_colors[mesh_name], label=mesh_name)
        
    plt.xlabel('h')
    plt.ylabel('pore_size/h')
    plt.title('Ratio of Pore Size to h for All Mesh Types')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()



# Function to get all possible values after lam, mue, gc, eps for strings starting with a given prefix
def get_values_for_prefix(strings_list, prefix):
    # Function to extract lam, mue, gc, eps values from a string
    def extract_values(string):
        # Regular expression pattern to capture lam, mue, Gc, eps values
        pattern = r"lam([0-9.]+)_mue([0-9.]+)_Gc([0-9.]+)_eps([0-9.]+)"
        match = re.search(pattern, string)
        
        if match:
            lam = float(match.group(1))
            mue = float(match.group(2))
            gc = float(match.group(3))
            eps = float(match.group(4))
            return lam, mue, gc, eps
        return None
    
    values = {'lam': set(), 'mue': set(), 'Gc': set(), 'eps': set()}  # Using sets to avoid duplicates
    
    for string in strings_list:
        if string.startswith(prefix):  # Check if the string starts with the given prefix
            result = extract_values(string)
            if result:
                lam, mue, gc, eps = result
                values['lam'].add(lam)
                values['mue'].add(mue)
                values['Gc'].add(gc)
                values['eps'].add(eps)
    
    # Convert sets to sorted lists for readability
    return {key: sorted(list(val)) for key, val in values.items()}

def merge_dicts(dicts_list):
    # Initialize a merged dictionary with empty sets for each key
    merged = {'lam': set(), 'mue': set(), 'Gc': set(), 'eps': set()}
    
    # Iterate over each dictionary in the input list
    for d in dicts_list:
        for key in merged:
            merged[key].update(d[key])  # Add the values from the current dictionary, avoiding duplicates
    
    # Convert sets to sorted lists for readability
    return {key: sorted(list(val)) for key, val in merged.items()}


def intersect_dicts(dict1, dict2):
    # Initialize a result dictionary for the intersection
    intersected = {'lam': [], 'mue': [], 'Gc': [], 'eps': []}
    
    # Loop over each key and find the intersection of values between the two dictionaries
    for key in intersected:
        intersected[key] = sorted(list(set(dict1[key]).intersection(set(dict2[key]))))
    
    return intersected


def read_all_simulation_data(base_path, graphs_filename = "run_simulation_graphs.txt", parameter_filename = "parameters.txt"):
    # List to store tuples of (data, parameters)
    simulation_results = []

    # Iterate through each item in the base directory
    for folder_name in os.listdir(base_path):
        # Check if the folder name starts with "simulation_"
        if folder_name.startswith("simulation_"):
            # Define paths for the data and parameter files
            data_path = os.path.join(base_path, folder_name, graphs_filename)
            parameter_path = os.path.join(base_path, folder_name, parameter_filename)
            
            # Try to read the data and parameter files
            try:
                # Read the simulation data
                data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)
                
                # Read the parameters
                parameters = pp.read_parameters_file(parameter_path)
                
                # Store the data and parameters together as a tuple
                simulation_results.append((data, parameters))
                
            except Exception as e:
                print(f"Error reading files in {folder_name}: {e}")
    
    return simulation_results


def plot_multiple_columns(data_objects, col_x, col_y, output_filename, 
                          vlines=None, hlines=None, xlabel=None, ylabel=None, 
                          title=None, legend_labels=None, 
                          xlabel_fontsize=24, ylabel_fontsize=24, title_fontsize=24, 
                          tick_fontsize=22, legend_fontsize=22, figsize=(10, 7), 
                          usetex=False, log_y=False, use_broad_palette=False,
                          x_range=None, y_range=None, use_bw_palette=True, show_markers=False):
    """
    Plots multiple datasets with the same x and y columns, allowing individual vertical and horizontal lines for each,
    with an optional logarithmic y-axis.

    Parameters:
        data_objects (list): List of data objects (DataFrames or dict-like) to be plotted.
        col_x (str): Column name for the x-axis.
        col_y (str): Column name for the y-axis.
        output_filename (str): File name to save the plot.
        vlines (list of lists): List of vertical line positions for each data object.
        hlines (list of lists): List of horizontal line positions for each data object.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        legend_labels (list): List of labels for the legend, corresponding to each data object.
        xlabel_fontsize (int): Font size for the x-axis label.
        ylabel_fontsize (int): Font size for the y-axis label.
        title_fontsize (int): Font size for the plot title.
        tick_fontsize (int): Font size for the axis tick labels.
        legend_fontsize (int): Font size for the legend labels.
        figsize (tuple): Figure dimensions as (width, height) in inches.
        usetex (bool): Whether to use LaTeX for rendering text in labels.
        log_y (bool): Whether to display the y-axis in logarithmic scale.
        use_broad_palette (bool): Whether to use a broad color palette or a narrow, distinguishable palette.
        x_range (tuple): Tuple specifying the x-axis range as (xmin, xmax). Default is no restriction.
        y_range (tuple): Tuple specifying the y-axis range as (ymin, ymax). Default is no restriction.
        use_bw_palette (bool): Whether to use a black-and-white palette with different line styles or shades of grey.
        show_markers (bool): Whether to display markers on the plot lines.
    """

    if usetex:
        plt.rcParams['text.usetex'] = True

    plt.figure(figsize=figsize)

    if use_bw_palette:
        # Define greys and line styles
        greys = ['black', 'dimgray', 'gray', 'darkgray', 'silver']
        linestyles = ['-', '--', '-.', ':']
        colors_and_styles = [(g, ls) for g in greys for ls in linestyles]
    elif use_broad_palette:
        colors = list(mcolors.CSS4_COLORS.values())  # Broad palette
        linestyles = ['-']
        colors_and_styles = [(c, '-') for c in colors]
    else:
        colors = plt.cm.tab10.colors  # Distinguishable narrow palette
        linestyles = ['-']
        colors_and_styles = [(c, '-') for c in colors]

    for i, data in enumerate(data_objects):
        color, linestyle = colors_and_styles[i % len(colors_and_styles)]

        plt.plot(data[col_x], data[col_y], marker='.' if show_markers else None, linestyle=linestyle, color=color, 
                 label=legend_labels[i] if legend_labels else f'Data {i+1}')

        if vlines and i < len(vlines):
            for vline in vlines[i]:
                plt.axvline(x=vline, color=color, linestyle='--', linewidth=0.5)

        if hlines and i < len(hlines):
            for hline in hlines[i]:
                plt.axhline(y=hline, color=color, linestyle='--', linewidth=0.5)

    ax = plt.gca()
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    plt.xlabel(xlabel if xlabel else f'Column {col_x}', fontsize=xlabel_fontsize)
    plt.ylabel(ylabel if ylabel else f'Column {col_y}', fontsize=ylabel_fontsize)
    plt.title(title if title else ' ', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    if not log_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved as {output_filename}")
    
    
def plot_multiple_lines(x_values, y_values, title='', x_label='', y_label='', 
                        legend_labels=None, output_file='plot.png', figsize=(10, 6), 
                        usetex=True, log_y=False, use_broad_palette=False, 
                        x_range=None, y_range=None, use_bw_palette=True, show_markers=False, 
                        xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=18, 
                        tick_fontsize=16, legend_fontsize=16):
    """
    Plots multiple lines on the same graph with additional customization options and saves the plot to a file.
    
    Parameters:
    - x_values: 2D list or numpy array containing x values for each line (shape: [n_lines, n_points]).
    - y_values: 2D list or numpy array containing y values for each line (shape: [n_lines, n_points]).
    - title: Title of the plot (default: '').
    - x_label: Label for the x-axis (default: '').
    - y_label: Label for the y-axis (default: '').
    - legend_labels: List of labels for each line in the legend (default: None).
    - output_file: File path (with extension) to save the plot (default: 'plot.png').
    - figsize: Figure dimensions as (width, height) in inches (default: (10, 6)).
    - usetex: Whether to use LaTeX for rendering text in labels (default: False).
    - log_y: Whether to display the y-axis in logarithmic scale (default: False).
    - use_broad_palette: Whether to use a broad color palette or a narrow, distinguishable palette (default: False).
    - x_range: Tuple specifying the x-axis range as (xmin, xmax) (default: None).
    - y_range: Tuple specifying the y-axis range as (ymin, ymax) (default: None).
    - use_bw_palette: Whether to use a black-and-white palette with different line styles or shades of grey (default: True).
    - show_markers: Whether to display markers on the plot lines (default: False).
    - xlabel_fontsize: Font size for the x-axis label (default: 18).
    - ylabel_fontsize: Font size for the y-axis label (default: 18).
    - title_fontsize: Font size for the plot title (default: 18).
    - tick_fontsize: Font size for the axis tick labels (default: 16).
    - legend_fontsize: Font size for the legend labels (default: 16).
    """
    
    if usetex:
        plt.rcParams['text.usetex'] = True

    plt.figure(figsize=figsize)

    # Choose palette and line styles based on user options
    if use_bw_palette:
        greys = ['black', 'dimgray', 'gray', 'darkgray', 'silver']
        linestyles = ['-', '--', '-.', ':']
        colors_and_styles = [(g, ls) for g in greys for ls in linestyles]
    elif use_broad_palette:
        colors = list(mcolors.CSS4_COLORS.values())  # Broad palette
        linestyles = ['-']
        colors_and_styles = [(c, '-') for c in colors]
    else:
        colors = plt.cm.tab10.colors  # Distinguishable narrow palette
        linestyles = ['-']
        colors_and_styles = [(c, '-') for c in colors]

    # Plot each line with custom color, style, and markers if requested
    for i in range(len(x_values)):
        color, linestyle = colors_and_styles[i % len(colors_and_styles)]
        plt.plot(x_values[i], y_values[i], marker='.' if show_markers else None, linestyle=linestyle, color=color, 
                 label=legend_labels[i] if legend_labels else f'Line {i+1}')

    # Optionally add x and y axis ranges
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Optionally set y-axis to logarithmic scale
    if log_y:
        ax = plt.gca()
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

    # Set labels and title with custom font sizes
    plt.xlabel(x_label, fontsize=xlabel_fontsize)
    plt.ylabel(y_label, fontsize=ylabel_fontsize)
    plt.title(title, fontsize=title_fontsize)
    
    # Add legend if there are labels
    plt.legend(fontsize=legend_fontsize)

    # Customize tick labels size
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Save the plot to the specified file
    plt.savefig(output_file)

    # Close the plot to free up memory
    plt.close()
    print(f"Plot saved as {output_file}")
    
    
def plot_single_line(x, y, filename, plot_type="line", title="", xlabel="X", ylabel="Y",
                   xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=16, tick_fontsize=16,
                   x_range=None, y_range=None, usetex=True, use_bw_palette=True, show_grid=False):
    """
    Plots x and y arrays and saves the plot to a file.

    Parameters:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - filename (str): The file path where the plot will be saved (e.g., 'plot.png').
    - plot_type (str): Type of plot ('line' for line plot, 'dot' for dot plot). Default is 'line'.
    - title (str): Title of the plot (default is "").
    - xlabel (str): Label for the x-axis (default is "X").
    - ylabel (str): Label for the y-axis (default is "Y").
    - xlabel_fontsize (int): Font size for the x-axis label (default is 14).
    - ylabel_fontsize (int): Font size for the y-axis label (default is 14).
    - title_fontsize (int): Font size for the plot title (default is 16).
    - tick_fontsize (int): Font size for the axis tick labels (default is 12).
    - x_range (tuple): Tuple specifying the x-axis range as (xmin, xmax). Default is no restriction.
    - y_range (tuple): Tuple specifying the y-axis range as (ymin, ymax). Default is no restriction.
    - usetex (bool): Whether to use LaTeX for rendering text in labels (default is False).
    - use_bw_palette (bool): Whether to use a black-and-white palette with black lines and no colors (default is False).
    - show_grid (bool): Whether to display the grid (default is True).
    """
    if usetex:
        plt.rcParams['text.usetex'] = True

    plt.figure(figsize=(8, 6))

    # Choose color and line style based on palette
    color = 'black' if use_bw_palette else 'blue'
    linestyle = '-' if plot_type == "line" else 'None'

    # Plot the data based on the selected plot type
    if plot_type == "dot":
        plt.scatter(x, y, label="Data", color=color)
    else:
        plt.plot(x, y, label="Data", color=color, linestyle=linestyle)

    # Set axis ranges if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Add title and labels
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)

    # Optional: Add grid
    if show_grid:
        plt.grid(True)

    # Set tick font sizes
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved as {filename}")
    
# def save_plot_data_to_file(x, y, filename):
#     """
#     Save x and y data arrays to a file.
    
#     Parameters:
#         x (array-like): The x values of the plot.
#         y (array-like): The y values of the plot.
#         filename (str): The name of the file to save the data to.
#     """
#     # Stack the x and y data into a single array and save to a file
#     data = np.column_stack((x, y))
#     np.savetxt(filename, data, header="x, y", delimiter=',', comments='')

def save_plot_data_to_file(x, y, filename):
    """
    Save x and y data arrays to a file, handling None entries in y data.

    Parameters:
        x (array-like): The x values of the plot.
        y (array-like): The y values of the plot.
        filename (str): The name of the file to save the data to.
    """
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Filter out entries where y is None
    valid_indices = [i for i, yi in enumerate(y) if yi is not None]
    x_filtered = x[valid_indices]
    y_filtered = y[valid_indices]

    # Stack the filtered x and y data into a single array and save to a file
    data = np.column_stack((x_filtered, y_filtered))
    np.savetxt(filename, data, header="x, y", delimiter=',', comments='')

def load_plot_data_from_file(filename):
    """
    Load x and y data from a file.
    
    Parameters:
        filename (str): The name of the file to load the data from.
    
    Returns:
        tuple: A tuple containing two arrays (x, y).
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Skip header
    x = data[:, 0]  # First column is x
    y = data[:, 1]  # Second column is y
    return x, y


def plot_multiple_columns_B(data_objects, col_x, col_y, output_filename, 
                          vlines=None, hlines=None, xlabel=None, ylabel=None, 
                          title=None, legend_labels=None, 
                          xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=18, 
                          tick_fontsize=16, legend_fontsize=16, figsize=(10, 6), 
                          usetex=False, log_y=False):
    """
    Plots multiple datasets with the same x and y columns, using shades of grey for line colors.
    
    Parameters:
        data_objects (list): List of data objects (DataFrames or dict-like) to be plotted.
        col_x (str): Column name for the x-axis.
        col_y (str): Column name for the y-axis.
        output_filename (str): File name to save the plot.
        vlines (list of lists): List of vertical line positions for each data object.
        hlines (list of lists): List of horizontal line positions for each data object.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        legend_labels (list): List of labels for the legend, corresponding to each data object.
        xlabel_fontsize (int): Font size for the x-axis label.
        ylabel_fontsize (int): Font size for the y-axis label.
        title_fontsize (int): Font size for the plot title.
        tick_fontsize (int): Font size for the axis tick labels.
        legend_fontsize (int): Font size for the legend labels.
        figsize (tuple): Figure dimensions as (width, height) in inches.
        usetex (bool): Whether to use LaTeX for rendering text in labels.
        log_y (bool): Whether to display the y-axis in logarithmic scale.
    """
    # Enable LaTeX rendering if requested
    if usetex:
        plt.rcParams['text.usetex'] = True

    # Set figure dimensions
    plt.figure(figsize=figsize)
    
    # Define a greyscale color palette
    greyscale_palette = ['black', 'dimgray', 'gray', 'darkgray', 'silver']
    
    for i, data in enumerate(data_objects):
        # Cycle through the greyscale palette for line colors
        color = greyscale_palette[i % len(greyscale_palette)]
        
        # Plot the data
        plt.plot(data[col_x], data[col_y], marker='.', linestyle='-', color=color, 
                 label=legend_labels[i] if legend_labels else f'Data {i+1}')
        
        # Add dashed vertical lines specific to this data object
        if vlines and i < len(vlines):
            for vline in vlines[i]:
                plt.axvline(x=vline, color=color, linestyle='--', linewidth=1)
        
        # Add dashed horizontal lines specific to this data object
        if hlines and i < len(hlines):
            for hline in hlines[i]:
                plt.axhline(y=hline, color=color, linestyle='--', linewidth=1)
    
    # Set the y-axis to logarithmic scale if requested
    ax = plt.gca()
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))  # Adds minor ticks for each order of magnitude
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels to avoid clutter
    
    # Set custom labels and title, with specific font sizes
    plt.xlabel(xlabel if xlabel else f'Column {col_x}', fontsize=xlabel_fontsize)
    plt.ylabel(ylabel if ylabel else f'Column {col_y}', fontsize=ylabel_fontsize)
    plt.title(title if title else f' ', fontsize=title_fontsize)
    
    # Add legend with custom font size
    plt.legend(fontsize=legend_fontsize)
    
    # Set the maximum number of ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit x-axis to 10 ticks
    if not log_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # For linear y-axis, limit to 10 ticks
    
    # Set fontsize for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Save the plot as a PNG file
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display in some environments
    print(f"Plot saved as {output_filename}") 
    
    
def compute_jump_size(x,y):
    #Calculates the jump size by locating a kink point first.
    # kink point is located by computing moving averages in intervals of size windowsize 
    # kink point is the beginning of interval where the moving average of dydx compared to the interval bording 
    # left on that interval decreases by 75% 
    # jump size is the difference between y at kink point and y at beginning of data y[0]
    
    merged_x = []
    merged_y = []

    # Loop through the data and merge repeated x-values
    for i in range(len(x)):
        if i == 0 or x[i] != x[i-1]:  # Keep the first occurrence of each x
            merged_x.append(x[i])
            merged_y.append(y[i])

    # Convert merged data back to numpy arrays
    merged_x = np.array(merged_x)
    merged_y = np.array(merged_y)
    
    dy = np.diff(merged_y)
    dx = np.diff(merged_x)

    # Calculate the normalized derivative (rate of change)
    dydx = dy / dx
    
    # Parameters
    window_size = 10
    threshold_drop = 0.5  # 50% drop

    # Compute moving averages of the derivative
    num_intervals = len(dydx) - window_size + 1
    # moving average interval n starts at x index n
    moving_averages = np.array([np.mean(dydx[i:i + window_size]) for i in range(num_intervals)])

    drop_index = None
    for i in range(window_size,len(moving_averages)):
        average_this_interval = moving_averages[i]
        average_left_interval = moving_averages[i-window_size]
        
        if average_this_interval < (0.5 * average_left_interval):
            drop_index = i
            break
    
    if drop_index is None:
        print("WARNING: No kink found")
        jump_size = None
    else:
        kink_index = drop_index
        jump_size = merged_y[kink_index] - merged_y[0]

    return jump_size

def get_initial_crack_length(find_max_y_under_x_threshold, data,x_threshold=None,y_col=9):
    if len(data.values) == 0:
        return 0.0
    default_x_threshold = data.values[0,0]+0.08
    if x_threshold is None:
        x_threshold = default_x_threshold
        
    A_initial_num = data.values[0,y_col]
    initial_crack_length = find_max_y_under_x_threshold(df=data, x_col=0, y_col=y_col, x_threshold=x_threshold)[1]
    initial_crack_length = initial_crack_length - A_initial_num
    
    # # TODO change
    # jump_size = compute_jump_size(data.values[:,0], data.values[:,y_col])
    # initial_crack_length = jump_size
    
    return initial_crack_length


def plot_columns(data, col_x, col_y, output_filename, vlines=None, hlines=None, 
                 xlabel=None, ylabel=None, title=None, 
                 xlabel_fontsize=18, ylabel_fontsize=18, title_fontsize=18, 
                 tick_fontsize=16, figsize=(10, 6), usetex=False, 
                 font_color="black", line_color="black", plot_dots=False, 
                 x_range=None, y_range=None):
    """
    Plots data from two specified columns with customization options.

    Parameters:
    - data: DataFrame containing the data to plot.
    - col_x: Column name for x-axis.
    - col_y: Column name for y-axis.
    - output_filename: Name of the file to save the plot.
    - vlines: List of x-coordinates for vertical lines to draw.
    - hlines: List of y-coordinates for horizontal lines to draw.
    - xlabel: Label for x-axis.
    - ylabel: Label for y-axis.
    - title: Title of the plot.
    - xlabel_fontsize, ylabel_fontsize, title_fontsize: Font sizes for respective labels and title.
    - tick_fontsize: Font size for ticks.
    - figsize: Tuple defining figure size.
    - usetex: Boolean to use LaTeX for text rendering.
    - font_color: Font color for labels and title.
    - line_color: Line color for the plot.
    - plot_dots: Boolean to toggle plotting dots on the line.
    - x_range: Tuple defining the range for the x-axis (min, max).
    - y_range: Tuple defining the range for the y-axis (min, max).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.rc('text', usetex=usetex)

    # Plot the data
    plt.plot(data[col_x], data[col_y], marker='.' if plot_dots else None, color=line_color, label=col_y)

    # Add vertical lines if specified
    if vlines:
        for vline in vlines:
            plt.axvline(x=vline, color='gray', linestyle='--', linewidth=1)

    # Add horizontal lines if specified
    if hlines:
        for hline in hlines:
            plt.axhline(y=hline, color='gray', linestyle='--', linewidth=1)

    # Set axis labels and title
    if xlabel:
        plt.xlabel(xlabel, fontsize=xlabel_fontsize, color=font_color)
    if ylabel:
        plt.ylabel(ylabel, fontsize=ylabel_fontsize, color=font_color)
    if title:
        plt.title(title, fontsize=title_fontsize, color=font_color)

    # Set axis ranges if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, labelcolor=font_color)

    # Add legend (if needed, uncomment below)
    # plt.legend()

    # Save the plot
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
