import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator

import alex.postprocessing as pp
import alex.homogenization as hom
import alex.linearelastic as le
import math
import alex.evaluation as ev


def find_simulation_by_wsteg(path, wsteg_value_in):
    # Iterate over all directories starting with "simulation_"
    for directory in os.listdir(path):
        dir_path = os.path.join(path, directory)
        
        # Check if it is a directory and starts with "simulation_"
        if os.path.isdir(dir_path) and directory.startswith("simulation_"):
            parameters_file = os.path.join(dir_path, "parameters.txt")
            
            # Check if parameters.txt exists in the directory
            if os.path.isfile(parameters_file):
                # Open and read the file line by line
                with open(parameters_file, "r") as file:
                    for line in file:
                        # Look for the line that starts with "wsteg="
                        if line.startswith("wsteg="):
                            # Extract the value and convert to float
                            try:
                                wsteg_value = float(line.split("=")[1].strip())
                                # Compare with the given value
                                if wsteg_value == wsteg_value_in:
                                    return dir_path
                            except ValueError:
                                continue  # Skip line if conversion fails

    # Return None if no directory with the matching wsteg value is found
    return None


# Define the path to the file based on the script directory
script_path = os.path.dirname(__file__)
# data_directory = os.path.join(script_path,'lam_mue_1.0_coarse')
# data_directory = os.path.join(script_path,'cubic_degrad')
data_directory = os.path.join(script_path,'results')


simulation_data_folder = find_simulation_by_wsteg(data_directory,wsteg_value_in=1.0)
data_path = os.path.join(simulation_data_folder, 'run_simulation_graphs.txt')
parameter_path = os.path.join(simulation_data_folder,"parameters.txt")

J_x_label = "$J_{x} / G_c$"

data = pd.read_csv(data_path, delim_whitespace=True, header=None, skiprows=1)

Jx_max = np.max(data[1])
print(f"Jx_max: {Jx_max}")

gc_num_quotient = 1.0
def normalize_Jx_to_Gc_num(gc_num_quotient, data):
    data.iloc[:, 1] = data.iloc[:, 1] / gc_num_quotient
normalize_Jx_to_Gc_num(gc_num_quotient, data)

output_file = os.path.join(script_path, 'PAPER_01_all_Jx_vs_xct_pf.png')
ev.plot_columns(data, 3, 1, output_file,xlabel="$x_{ct} / L$",ylabel=J_x_label, usetex=True, title=" ", plot_dots=False, y_range=[0.0,1.2])


