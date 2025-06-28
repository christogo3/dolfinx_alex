import os
import pandas as pd
import alex.evaluation as ev

# Define J_c scaling factors
jc_scale = 0.371
jc_scale_2 = 0.7346

# Get script path
script_path = os.path.dirname(__file__)

# Load datasets
path_ro = os.path.join(script_path, 'run_simulation_ramberg_osgood_graphs.txt')
data_ramberg_osgood = pd.read_csv(path_ro, delim_whitespace=True, header=None, skiprows=1)

path_gc1 = os.path.join(script_path, 'run_simulation_linear_elastic_graphs_gc1.0.txt')
data_linear_elastic_gc1 = pd.read_csv(path_gc1, delim_whitespace=True, header=None, skiprows=1)

path_gc_scaled = os.path.join(script_path, 'run_simulation_linear_elastic_graphs0.5679.txt')
data_linear_elastic_gc_scaled = pd.read_csv(path_gc_scaled, delim_whitespace=True, header=None, skiprows=1)

path_gc_scaled_2 = os.path.join(script_path, 'run_simulation_linear_elastic_graphs.txt')
data_linear_elastic_gc_scaled_2 = pd.read_csv(path_gc_scaled_2, delim_whitespace=True, header=None, skiprows=1)

# Compute maxima
max_ro = data_ramberg_osgood[1].max()
max_gc1 = data_linear_elastic_gc1[1].max()
max_gc_scaled = data_linear_elastic_gc_scaled[1].max()
max_gc_scaled_2 = data_linear_elastic_gc_scaled_2[1].max()

max_pi_ro = data_ramberg_osgood[2].max()
max_pi_gc1 = data_linear_elastic_gc1[2].max()
max_pi_gc_scaled = data_linear_elastic_gc_scaled[2].max()
max_pi_gc_scaled_2 = data_linear_elastic_gc_scaled_2[2].max()

# Labels (aligned with second script)
label_ro = r"\textbf{RO} (max $\sigma^*$ $\approx$ " + f"{max_ro:.2f}" + r"$\sigma_y$)"
label_gc1 = r"\textbf{Eq$\mathbf{J_c}$} (max $\sigma^*$ $\approx$ " + f"{max_gc1:.2f}" + r"$\sigma_y$)"
label_gc_scaled = r"\textbf{Eq$\mathbf{\sigma^*}$} (max $\sigma^*$ $\approx$ " + f"{max_gc_scaled:.2f}" + r"$\sigma_y$)"
label_gc_scaled_2 = r"\textbf{Eq$\mathbf{\Pi^*}$} (max $\sigma^*$ $\approx$ " + f"{max_gc_scaled_2:.2f}" + r"$\sigma_y$)"

label_pi_ro = r"\textbf{RO} (max $\Pi^*$ $\approx$ " + f"{max_pi_ro:.2f}" + r"$J_c^0L$)"
label_pi_gc1 = r"\textbf{Eq$\mathbf{J_c}$} (max $\Pi^*$ $\approx$ " + f"{max_pi_gc1:.2f}" + r"$J_c^0L$)"
label_pi_gc_scaled = r"\textbf{Eq$\mathbf{\sigma^*}$} (max $\Pi^*$ $\approx$ " + f"{max_pi_gc_scaled:.2f}" + r"$J_c^0L$)"
label_pi_gc_scaled_2 = r"\textbf{Eq$\mathbf{\Pi^*}$} (max $\Pi^*$ $\approx$ " + f"{max_pi_gc_scaled_2:.2f}" + r"$J_c^0L$)"

# Output files
output_file_stress = os.path.join(script_path, 'PAPER_reaction_forces_1D.png')
output_file_energy = os.path.join(script_path, 'PAPER_energy_vs_displacement_1D.png')

# Plot stress
ev.plot_multiple_columns(
    [data_ramberg_osgood, data_linear_elastic_gc1, data_linear_elastic_gc_scaled, data_linear_elastic_gc_scaled_2],
    0, 1,
    output_file_stress,
    legend_labels=[label_ro, label_gc1, label_gc_scaled, label_gc_scaled_2],
    usetex=True,
    xlabel=r"$\varepsilon / \sqrt{J_c^{0} / (L\mu_0)}$",
    ylabel=r"$\sigma / \sigma_y$",
    y_range=[0.0, 2.0],
    x_range=[0.0, 1.25],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10,8)
)

# Plot energy
ev.plot_multiple_columns(
    [data_ramberg_osgood, data_linear_elastic_gc1, data_linear_elastic_gc_scaled, data_linear_elastic_gc_scaled_2],
    0, 2,
    output_file_energy,
    legend_labels=[label_pi_ro, label_pi_gc1, label_pi_gc_scaled, label_pi_gc_scaled_2],
    usetex=True,
    xlabel=r"$u_{0} / \sqrt{J_c^{0}L / \mu_0}$",
    ylabel=r"$\Pi / J_c^0L$",
    y_range=None,
    x_range=[0.0, 1.25],
    markers_only=False,
    marker_size=4,
    use_colors=True,
    legend_fontsize=20,
    figsize=(10,8)
)




