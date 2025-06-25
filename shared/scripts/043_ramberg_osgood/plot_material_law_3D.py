import numpy as np
import pandas as pd
import os
import alex.evaluation as ev  # plot_multiple_columns is assumed to be here

# === Material Parameters ===
mu0 = 1.0       # Initial shear modulus
lam0 = 1.0      # Initial Lamé parameter
K = lam0 + 2.0 * mu0 / 3.0  # Bulk modulus
sigma_y = 1.0   # Yield stress
r = 10          # Smoothness
b_values = [0.1, 0.5, 1.0]  # Material law parameter

# === Poisson's ratio ===
nu0 = lam0 / (2 * (lam0 + mu0))

# === Strain range ===
eps0_vals = np.linspace(0.0, 1.0, 500)

# === Utility functions ===
def strain_tensor(eps0):
    return np.array([
        [eps0, 0, 0],
        [0, -nu0 * eps0, 0],
        [0, 0, -nu0 * eps0]
    ])

def deviatoric(tensor):
    return tensor - np.trace(tensor)/3.0 * np.eye(3)

def tensor_norm(tensor):
    return np.sqrt(np.sum(tensor**2))

def mu_eps(e_eq, mu0, b, sigma_y, r):
    arg = (3/2) * 2 * mu0 * e_eq / sigma_y
    return mu0 * (b + (1 - b) / (1 + abs(arg)**r)**(1/r))

def stress_tensor(eps_tensor, mu_func, b, sigma_y, r):
    e_dev = deviatoric(eps_tensor)
    e_eq = np.sqrt(2/3 * np.sum(e_dev**2))
    mu_val = mu_func(e_eq, mu0, b, sigma_y, r)
    return K * np.trace(eps_tensor) * np.eye(3) + 2 * mu_val * e_dev

# === Output containers ===
data_sig11 = []
data_devnorm = []
legend_labels = []

# === Loop over b-values ===
for b in b_values:
    sig11_list = []
    e0_list = []
    s_dev_norm_list = []
    e_dev_norm_list = []

    for eps0 in eps0_vals:
        eps_tensor = strain_tensor(eps0)
        e_dev = deviatoric(eps_tensor)
        e_eq = np.sqrt(2/3 * np.sum(e_dev**2))
        mu_val = mu_eps(e_eq, mu0, b, sigma_y, r)
        sig_tensor = stress_tensor(eps_tensor, mu_eps, b, sigma_y, r)
        s_dev = deviatoric(sig_tensor)

        sig11_list.append(sig_tensor[0, 0])
        e0_list.append(eps0)
        s_dev_norm_list.append(tensor_norm(s_dev))
        e_dev_norm_list.append(tensor_norm(e_dev))

    # Create DataFrames
    df1 = pd.DataFrame({0: e0_list, 1: sig11_list})
    df2 = pd.DataFrame({0: e_dev_norm_list, 1: s_dev_norm_list})
    data_sig11.append(df1)
    data_devnorm.append(df2)
    legend_labels.append(fr"$b={b}$")

# === Plot settings ===
script_path = os.path.dirname(__file__)
output_sig11 = os.path.join(script_path, 'PAPER_3D_material_law_sig11_vs_eps0.png')
output_devnorm = os.path.join(script_path, 'PAPER_3D_material_law_devnorm.png')

# === Plot σ₁₁ vs ε₀ ===
ev.plot_multiple_columns(
    data_sig11,
    col_x=0,
    col_y=1,
    output_filename=output_sig11,
    xlabel=r"$\varepsilon_0$",
    ylabel=r"$\sigma_{11}$ / $\sigma_y$",
    # title=r"Uniaxial Stress–Strain Response (3D)",
    hlines=[[sigma_y]],
    legend_labels=legend_labels,
    x_range=[0, 1.0],
    y_range=[0, 2.5],
    use_colors=True,
    usetex=True,
    # legend_fontsize=18,
    marker_size=4,
)

# === Plot ‖s‖ vs ‖e‖ ===
ev.plot_multiple_columns(
    data_devnorm,
    col_x=0,
    col_y=1,
    output_filename=output_devnorm,
    hlines=[[np.sqrt(2/3)*sigma_y]],
    xlabel=r"$\|{e}\|$",
    ylabel=r"$\|{s}\|$ / $\sigma_y$",
    # title=r"Deviatoric Stress–Strain Norms (3D)",
    legend_labels=legend_labels,
    x_range=[0, 1.0],
    y_range=[0, 2.5],
    use_colors=True,
    usetex=True,
    # legend_fontsize=18,
    marker_size=4,
)
