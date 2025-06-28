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
from scipy.signal import savgol_filter


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

# === Axis Labels ===
t_label = r"$t / [ L / {\dot{x}}_{\mathrm{bc}} ]$"
crack_tip_position_label = r"$x_{\mathrm{ct}}$"
J_x_label = r"$J_{x} / J_c^0$"

# === Setup ===
script_path = os.path.dirname(__file__)
gc_num_quotient = 1.0

def normalize_Jx_to_Gc_num(gc_num_quotient, data):
    data.iloc[:, 1] = data.iloc[:, 1] / gc_num_quotient

# === Ramberg-Osgood Data ===
data_ramberg_osgood = pd.read_csv(
    os.path.join(script_path, 'results', 'simulation_20250625_184944', 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)
smoothed_col1 = savgol_filter(data_ramberg_osgood[1], window_length=200, polyorder=5)
# Replace column 1 with the smoothed version
data_ramberg_osgood[1] = smoothed_col1


normalize_Jx_to_Gc_num(gc_num_quotient, data_ramberg_osgood)
Jx_max_ramberg_osgood = np.max(data_ramberg_osgood[1])
print(f"Jx_max Ramberg Osgood: {Jx_max_ramberg_osgood}")

ramberg_osgood_prefix = r"\textbf{RO} (max $J_x$ $\approx$ "
ramberg_osgood_value = f"{Jx_max_ramberg_osgood:.2f}"
ramberg_osgood_suffix = r"$J_c^0$)"
ramberg_osgood_label = ramberg_osgood_prefix + ramberg_osgood_value + ramberg_osgood_suffix

# === Linear Elastic Dataset 1 ===
linear_elastic_1 = pd.read_csv(
    os.path.join(script_path, '..', '049-J-Integral-wo-pores-deg-cubic', 'results', 'simulation_20250624_193413_gc1.0', 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)
normalize_Jx_to_Gc_num(gc_num_quotient, linear_elastic_1)
Jx_max_1 = np.max(linear_elastic_1[1])
print(f"Jx_max linear elastic 1: {Jx_max_1}")

label_1_prefix = r"\textbf{Eq$\mathbf{J_c}$} (max $J_x$ $\approx$ "
label_1_value = f"{Jx_max_1:.2f}"
label_1_suffix = r"$J_c^0$)"
label_1 = label_1_prefix + label_1_value + label_1_suffix

# === Linear Elastic Dataset 2 ===
linear_elastic_2 = pd.read_csv(
    os.path.join(script_path, '..', '049-J-Integral-wo-pores-deg-cubic', 'results', 'simulation_20250626_205832_0.371', 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)
normalize_Jx_to_Gc_num(gc_num_quotient, linear_elastic_2)
Jx_max_2 = np.max(linear_elastic_2[1])
print(f"Jx_max linear elastic 2: {Jx_max_2}")

label_2_prefix = r"\textbf{Eq$\mathbf{\sigma^*}$} (max $J_x$ $\approx$ "
label_2_value = f"{Jx_max_2:.2f}"
label_2_suffix = r"$J_c^0$)"
label_2 = label_2_prefix + label_2_value + label_2_suffix

# === Linear Elastic Dataset 3 ===
linear_elastic_3 = pd.read_csv(
    os.path.join(script_path, '..', '049-J-Integral-wo-pores-deg-cubic', 'results', 'simulation_20250626_215527_0.7346', 'run_simulation_graphs.txt'),
    delim_whitespace=True, header=None, skiprows=1
)
normalize_Jx_to_Gc_num(gc_num_quotient, linear_elastic_3)
Jx_max_3 = np.max(linear_elastic_3[1])
print(f"Jx_max linear elastic 3: {Jx_max_3}")

label_3_prefix = r"\textbf{Eq$\mathbf{\Pi^*}$} (max $J_x$ $\approx$ "
label_3_value = f"{Jx_max_3:.2f}"
label_3_suffix = r"$J_c^0$)"
label_3 = label_3_prefix + label_3_value + label_3_suffix

# === Legend Labels ===
legend_labels = [ramberg_osgood_label, label_1, label_2, label_3]

# === Plot J_x vs x_ct ===
output_file = os.path.join(script_path, 'PAPER_01_all_Jx_vs_xct_pf_2.png')
ev.plot_multiple_columns(
    [data_ramberg_osgood, linear_elastic_1, linear_elastic_2, linear_elastic_3],
    3, 1,
    output_file,
    legend_labels=legend_labels,
    usetex=True,
    xlabel=r"$x_{ct} / L$",
    ylabel=J_x_label,
    y_range=[-1.0, 1.5],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20
)

# === Plot crack tip position vs time ===
output_file = os.path.join(script_path, 'PAPER_00_xct_pf_vs_xct_KI_griffith.png')
ev.plot_columns_multiple_y(
    data=data_ramberg_osgood,
    col_x=0,
    col_y_list=[3, 4],
    output_filename=output_file,
    legend_labels=[crack_tip_position_label, r"$x_{\mathrm{bc}}$"],
    usetex=True,
    title=" ",
    plot_dots=False,
    xlabel=t_label,
    ylabel=r"crack tip position $/ L$",
    x_range=[-0.1, 20]
)


