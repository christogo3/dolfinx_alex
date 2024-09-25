import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import re
from collections import defaultdict
from typing import Callable, List, Dict, Tuple


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
    h_all_list=None
):
    # Define color palettes for different datasets
    color_palettes = [
        {"fine_pores": mcolors.to_rgba('darkred'), "medium_pores": mcolors.to_rgba('red'), "coarse_pores": mcolors.to_rgba('lightcoral')}, # Reds
        {"fine_pores": mcolors.to_rgba('darkblue'), "medium_pores": mcolors.to_rgba('blue'), "coarse_pores": mcolors.to_rgba('lightblue')}, # Blues
        {"fine_pores": mcolors.to_rgba('darkgreen'), "medium_pores": mcolors.to_rgba('green'), "coarse_pores": mcolors.to_rgba('lightgreen')}, # Greens
        {"fine_pores": mcolors.to_rgba('darkorange'), "medium_pores": mcolors.to_rgba('orange'), "coarse_pores": mcolors.to_rgba('lightsalmon')}, # Oranges
        # Add more color palettes if needed for additional datasets
    ]

    plt.figure(figsize=(10, 20))
    unique_labels = set()

    # Iterate over each results_dict and corresponding keys
    for i, (results_dict, keys) in enumerate(zip(results_dicts, keys_list)):
        # Create the max dictionary for the current results_dict
        max_Jx_dict = create_max_dict(results_dict, column_index=1)
        
        # Initialize the eps_pore_size_ratio values for each key
        pore_size_eps_ratio_values = []

        # Compute eps_pore_size_ratio for all keys in the current max_Jx_dict
        for key in keys:
            params = extract_parameters(key)
            if params:
                mesh_type = params[0]
                eps_value = get_eps(reference_L_global, params[4])
                pore_size_eps_ratio = get_pore_size_eps_ratio(mesh_type, eps_value, pore_size_all)
                pore_size_eps_ratio_values.append(pore_size_eps_ratio)
        
        # Select the color palette for the current dataset
        mesh_colors = color_palettes[i % len(color_palettes)]

        # Marker types based on (lam, mue) values
        marker_types = {
            (1.0, 1.0): 'o',      # Circle
            (1.5, 1.0): 'p',      # Cross
            (0.6667, 1.0): 'v',      # Dot
            (10.0, 10.0): '^',    # Triangle
            (15.0, 10.0): 's'     # Square
        }

        # Plot max_Jx_dict values vs eps_pore_size_ratio_values with colors and markers
        for key, pore_size_eps_ratio in zip(keys, pore_size_eps_ratio_values):
            params = extract_parameters(key)
            if params:
                mesh_type = params[0]
                lam_mue = (params[1], params[2])
                
                color = mesh_colors.get(mesh_type, 'black')
                marker = marker_types.get(lam_mue, 'x')  # Default marker is 'x' if (lam, mue) is not in the marker_types dictionary
                label = f"{mesh_type} (Set {i+1})" if mesh_type not in unique_labels else ""
                
                # Check if the current key is in the special_keys set
                if special_keys_list and key in special_keys_list[i]:
                    edge_color = 'black'
                    linewidth = 2.0
                else:
                    edge_color = 'none'
                    linewidth = 0
                
                y_value = max_Jx_dict[key] / get_gc_num_for_key(key, h_all=h_all_list[i], reference_L_global=reference_L_global)
                plt.scatter(pore_size_eps_ratio, y_value, color=color, marker=marker, s=100, label=label, edgecolor=edge_color, linewidth=linewidth)
                unique_labels.add(mesh_type)
                
                # Display Gc and inverse of eps values as tuples to the right of the markers
                plt.text(pore_size_eps_ratio, y_value, f"({params[3]}, {pore_size_eps_ratio})", fontsize=9, ha='left', va='bottom')

    # Create custom legend handles
    handles = []
    
    # Add mesh type legend handles for each results_dict
    for i, _ in enumerate(results_dicts):
        mesh_colors = color_palettes[i % len(color_palettes)]
        for mesh_type, color in mesh_colors.items():
            handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=12, label=f'{mesh_type} (Set {i+1})'))
    
    # Add marker type legend handles with Poisson ratio included
    for (lam, mue), marker in marker_types.items():
        nu = poisson_ratio(lam, mue)
        handles.append(mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=12, label=f'lam={lam}, mue={mue} (nu={nu:.2f})'))
    
    plt.xlabel('pore_size_eps_ratio')
    plt.ylabel('Max Jx / Gc_num')
    plt.title(plot_title)
    plt.legend(handles=handles, title="Legend", loc='best')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
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