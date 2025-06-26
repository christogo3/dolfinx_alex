import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
import matplotlib.colors as mcolors
import os
import numpy as np
import math

import alex.postprocessing as pp
import alex.homogenization as hom
import alex.linearelastic as le
import alex.evaluation as ev


def find_simulation_by_wsteg(path, wsteg_value_in):
    for directory in os.listdir(path):
        dir_path = os.path.join(path, directory)
        if os.path.isdir(dir_path) and directory.startswith("simulation_"):
            parameters_file = os.path.join(dir_path, "parameters.txt")
            if os.path.isfile(parameters_file):
                with open(parameters_file, "r") as file:
                    for line in file:
                        if line.startswith("wsteg="):
                            try:
                                wsteg_value = float(line.split("=")[1].strip())
                                if wsteg_value == wsteg_value_in:
                                    return dir_path
                            except ValueError:
                                continue
    return None


t_label = "$t / [ L / {\dot{x}}_{\mathrm{bc}} ]$"
crack_tip_position_label = "$x_{\mathrm{ct}}$"

# Define script and data paths
script_path = os.path.dirname(__file__)
data_directory_ramberg_osgood = os.path.join(script_path, 'results')

# Ramberg-Osgood data
data_path_ramberg_osgood = os.path.join(
    script_path,
    'results',
    'simulation_20250625_184944',
    'run_simulation_graphs.txt'
)
ramberg_osgood_label = "Ramberg Osgood"
data_ramberg_osgood = pd.read_csv(data_path_ramberg_osgood, delim_whitespace=True, header=None, skiprows=1)

# Compute max Jx and normalize
Jx_max_ramberg_osgood = np.max(data_ramberg_osgood[1])
print(f"Jx_max Ramberg Osgood: {Jx_max_ramberg_osgood}")

gc_num_quotient = 1.0
def normalize_Jx_to_Gc_num(gc_num_quotient, data):
    data.iloc[:, 1] = data.iloc[:, 1] / gc_num_quotient

normalize_Jx_to_Gc_num(gc_num_quotient, data_ramberg_osgood)

# Linear Elastic data
data_path_linear_elastic_gc10 = os.path.join(
    script_path,
    '..',
    '049-J-Integral-wo-pores-deg-cubic',
    'results',
    'simulation_20250624_193413_gc1.0',
    'run_simulation_graphs.txt'
)
linear_elastic_label_gc10 = "linear elastic $J_c=1.0J_c^0$"
data_linear_elastic_gc10 = pd.read_csv(data_path_linear_elastic_gc10, delim_whitespace=True, header=None, skiprows=1)

Jx_max_linear_elastic = np.max(data_linear_elastic_gc10[1])
print(f"Jx_max linear elastic: {Jx_max_linear_elastic}")


# Linear Elastic data
data_path_linear_elastic_gc05679 = os.path.join(
    script_path,
    '..',
    '049-J-Integral-wo-pores-deg-cubic',
    'results',
    'simulation_20250625_133139_gc0.5679',
    'run_simulation_graphs.txt'
)
linear_elastic_label_gc05679 = "linear elastic $J_c=0.5679J_c^0$"
data_linear_elastic_gc05679 = pd.read_csv(data_path_linear_elastic_gc05679, delim_whitespace=True, header=None, skiprows=1)

Jx_max_linear_elastic_gc05679 = np.max(data_linear_elastic_gc05679[1])
print(f"Jx_max linear elastic reduced: {Jx_max_linear_elastic_gc05679}")

# Create dynamic legend labels with max Jx
legend_labels = [
    f"{ramberg_osgood_label} (max $J_x$ = {Jx_max_ramberg_osgood:.2f}$J_c^0$)",
    f"{linear_elastic_label_gc10} (max $J_x$ = {Jx_max_linear_elastic:.2f}$J_c^0$)",
    f"{linear_elastic_label_gc05679} (max $J_x$ = {Jx_max_linear_elastic_gc05679:.2f}$J_c^0$)"
]

# Output file path
output_file = os.path.join(script_path, 'PAPER_01_all_Jx_vs_xct_pf_2.png')

J_x_label = "$J_{x} / J_c^0$"

# Plot
ev.plot_multiple_columns(
    [data_ramberg_osgood, data_linear_elastic_gc10, data_linear_elastic_gc05679],
    3, 1,
    output_file,
    legend_labels=legend_labels,
    usetex=True,
    xlabel="$x_{ct} / L$",
    ylabel=J_x_label,
    y_range=[0.0, 1.5],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20
)




output_file = os.path.join(script_path, 'PAPER_00_xct_pf_vs_xct_KI_griffith.png')  
ev.plot_columns_multiple_y(data=data_ramberg_osgood,col_x=0,col_y_list=[3,4],output_filename=output_file,
                        legend_labels=[crack_tip_position_label, "$x_{\mathrm{bc}}$"],usetex=True, title=" ", plot_dots=False,
                        xlabel=  t_label,ylabel="crack tip position"+" $/ L$",
                        x_range=[-0.1, 20],
                        # vlines=[hole_positions_out, hole_positions_out]
                        )