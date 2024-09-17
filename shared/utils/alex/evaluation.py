import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os


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
    return gc * (1.0 + h_value / eps / 4.0)

def get_eps(reference_L_global, eps_factor):
    eps = reference_L_global / eps_factor
    return eps

def get_eps_pore_size_ratio(mesh_name, eps_value, pore_size_all):
    pore_size = pore_size_all[mesh_name]
    ratio = pore_size / eps_value
    return ratio



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