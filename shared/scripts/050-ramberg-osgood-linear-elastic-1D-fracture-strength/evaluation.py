import os
import pandas as pd
import alex.evaluation as ev

# Define J_c scaling factor as a parameter
jc_scale = 0.371  # Replace this number with any other to change dynamically

# Get the directory path of this script
script_path = os.path.dirname(__file__)

# Load data
path_gc1 = os.path.join(script_path, 'run_simulation_linear_elastic_graphs_gc1.0.txt')
data_linear_elastic_gc1 = pd.read_csv(path_gc1, delim_whitespace=True, header=None, skiprows=1)

path_gc_scaled = os.path.join(script_path, 'run_simulation_linear_elastic_graphs.txt')
data_linear_elastic_gc_scaled = pd.read_csv(path_gc_scaled, delim_whitespace=True, header=None, skiprows=1)

path_ro = os.path.join(script_path, 'run_simulation_ramberg_osgood_graphs.txt')
data_ramberg_osgood = pd.read_csv(path_ro, delim_whitespace=True, header=None, skiprows=1)

# Compute max values for stress (column 1)
max_gc1 = data_linear_elastic_gc1[1].max()
max_gc_scaled = data_linear_elastic_gc_scaled[1].max()
max_ro = data_ramberg_osgood[1].max()

# Compute max values for energy (column 2)
max_pi_gc1 = data_linear_elastic_gc1[2].max()
max_pi_gc_scaled = data_linear_elastic_gc_scaled[2].max()
max_pi_ro = data_ramberg_osgood[2].max()

# Print max values
print(f"Maximum stress (column 1) - linear elastic $J_c=1.0J_c^{{0}}$: {max_gc1:.4f}")
print(f"Maximum stress (column 1) - linear elastic $J_c={jc_scale}J_c^{{0}}$: {max_gc_scaled:.4f}")
print(f"Maximum stress (column 1) - Ramberg-Osgood $J_c=1.0J_c^{{0}}$: {max_ro:.4f}")

print(f"Maximum $\Pi$ (column 2) - linear elastic $J_c=1.0J_c^{{0}}$: {max_pi_gc1:.4f}")
print(f"Maximum $\Pi$ (column 2) - linear elastic $J_c={jc_scale}J_c^{{0}}$: {max_pi_gc_scaled:.4f}")
print(f"Maximum $\Pi$ (column 2) - Ramberg-Osgood $J_c=1.0J_c^{{0}}$: {max_pi_ro:.4f}")

# Labels for stress plot (column 1)
label_gc1 = f"linear elastic $J_c=1.0J_c^{{0}}$,  $\sigma^{{*}}={max_gc1:.2f}$ / $\sigma_y$"
label_gc_scaled = f"linear elastic $J_c={jc_scale}J_c^{{0}}$,  $\sigma^{{*}}={max_gc_scaled:.2f}$ / $\sigma_y$"
label_ro = f"Ramberg Osgood $J_c=1.0J_c^{{0}}$,  $\sigma^{{*}}={max_ro:.2f}$ / $\sigma_y$"

# Labels for energy plot (column 2)
label_pi_gc1 = f"linear elastic $J_c=1.0J_c^{{0}}$,  $\Pi^{{*}}={max_pi_gc1:.2f}/ ( J_c^0L)$"
label_pi_gc_scaled = f"linear elastic $J_c={jc_scale}J_c^{{0}}$,  $\Pi^{{*}}={max_pi_gc_scaled:.2f}/ ( J_c^0L)$"
label_pi_ro = f"Ramberg Osgood $J_c=1.0J_c^{{0}}$,  $\Pi^{{*}}={max_pi_ro:.2f}/ ( J_c^0L)$"

# Output file paths
output_file_stress = os.path.join(script_path, 'PAPER_reaction_forces_1D.png')
output_file_energy = os.path.join(script_path, 'PAPER_energy_vs_displacement_1D.png')

# Plot stress vs. displacement
ev.plot_multiple_columns(
    [data_linear_elastic_gc1, data_linear_elastic_gc_scaled, data_ramberg_osgood],
    0, 1,
    output_file_stress,
    legend_labels=[label_gc1, label_gc_scaled, label_ro],
    usetex=True,
    xlabel="$u_{0} / \sqrt{J_c^{0}{L} / (\mu_0)}$",
    ylabel="$\sigma$ / $\sigma_y$",
    y_range=[0.0, 2.0],
    x_range=[0.0, 1.25],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10,8)
)

# Plot energy vs. displacement
ev.plot_multiple_columns(
    [data_linear_elastic_gc1, data_linear_elastic_gc_scaled, data_ramberg_osgood],
    0, 2,
    output_file_energy,
    legend_labels=[label_pi_gc1, label_pi_gc_scaled, label_pi_ro],
    usetex=True,
    xlabel="$u_{0} / \sqrt{J_c^{0}{L} / (\mu_0)}$",
    ylabel="$\Pi / ( J_c^0L)$",
    y_range=None,
    x_range=[0.0, 1.25],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10,8)
)


