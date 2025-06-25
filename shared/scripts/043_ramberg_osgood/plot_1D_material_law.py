import numpy as np
import pandas as pd
import os
import alex.evaluation as ev  # Assumes plot_multiple_columns is here

# === Material Parameters ===
E0 = 2.0        # Reference Young's modulus
sigma_y = 1.0   # Yield stress
r = 10          # Transition smoothness
b_values = [0.1, 0.5, 1.0]  # Different material law parameter values

# === Strain Range ===
strain_vals = np.linspace(0, 1.0, 500)

# === Define E(eps) and sigma(eps) ===
def E_eps(eps, E0, b, sigma_y, r):
    arg = np.abs(E0 * eps / sigma_y)
    return E0 * (b + (1 - b) / (1 + arg**r)**(1/r))

def sigma_eps(eps, E0, b, sigma_y, r):
    return E_eps(eps, E0, b, sigma_y, r) * eps

# === Generate Data for Each b ===
data_objects = []
legend_labels = []

for b in b_values:
    stress_vals = sigma_eps(strain_vals, E0, b, sigma_y, r)
    data = pd.DataFrame({0: strain_vals, 1: stress_vals})
    data_objects.append(data)
    legend_labels.append(fr"$b={b}$")

# === Output Path ===
script_path = os.path.dirname(__file__)
output_file = os.path.join(script_path, 'PAPER_1D_material_law_multiple_b.png')

# === Plot ===
ev.plot_multiple_columns(
    data_objects,
    col_x=0,
    col_y=1,
    output_filename=output_file,
    xlabel=r"$\varepsilon$",
    ylabel=r"$\sigma(\varepsilon)$ / $\sigma_y$",
    # title=r"Stress–Strain Law for Different $b$ Values",
    legend_labels=legend_labels,
    hlines=[[1.0]],
    x_range=[0, 1.0],
    y_range=[0, 2.0],
    use_colors=True,
    usetex=True,
    show_markers=False,
    markers_only=False,
    marker_size=5,
    # legend_fontsize=20
)
