import os
import pandas as pd
import alex.evaluation as ev

# Get the directory path of this script
script_path = os.path.dirname(__file__)

# Load data
path_gc1 = os.path.join(script_path, 'run_simulation_linear_elastic_graphs_gc1.0.txt')
data_linear_elastic_gc1 = pd.read_csv(path_gc1, delim_whitespace=True, header=None, skiprows=1)

path_gc05679 = os.path.join(script_path, 'run_simulation_linear_elastic_graphs_gc0.5679.txt')
data_linear_elastic_gc05679 = pd.read_csv(path_gc05679, delim_whitespace=True, header=None, skiprows=1)

path_ro = os.path.join(script_path, 'run_simulation_ramberg_osgood_graphs.txt')
data_ramberg_osgood = pd.read_csv(path_ro, delim_whitespace=True, header=None, skiprows=1)

# Compute max values
max_gc1 = data_linear_elastic_gc1[1].max()
max_gc05679 = data_linear_elastic_gc05679[1].max()
max_ro = data_ramberg_osgood[1].max()

# Print max values
print(f"Maximum stress (column 1) - linear elastic $J_c=1.0J_c^{{0}}$: {max_gc1:.4f}")
print(f"Maximum stress (column 1) - linear elastic $J_c=0.5679J_c^{{0}}$: {max_gc05679:.4f}")
print(f"Maximum stress (column 1) - Ramberg-Osgood $J_c=1.0J_c^{{0}}$: {max_ro:.4f}")

# Labels including max values for legend
label_gc1 = f"linear elastic $J_c=1.0J_c^{{0}}$,  $\sigma^{{*}}={max_gc1:.2f}$"
label_gc05679 = f"linear elastic $J_c=0.5679J_c^{{0}}$,  $\sigma^{{*}}={max_gc05679:.2f}$"
label_ro = f"Ramberg Osgood $J_c=1.0J_c^{{0}}$,  $\sigma^{{*}}={max_ro:.2f}$"

# Output plot path
output_file = os.path.join(script_path, 'reaction_forces_1D.png')

# Plot the data
ev.plot_multiple_columns(
    [data_linear_elastic_gc1, data_linear_elastic_gc05679, data_ramberg_osgood],
    0, 1,
    output_file,
    legend_labels=[label_gc1, label_gc05679, label_ro],
    usetex=True,
    xlabel="$u_{0} / L$",
    ylabel="$\sigma$ / $[\sqrt{{2\mu J_c} / L}]$",
    y_range=[0.0, 2.0],
    x_range=[0.0, 1.25],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20
)

