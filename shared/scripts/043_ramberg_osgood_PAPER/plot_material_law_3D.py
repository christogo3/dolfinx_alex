import numpy as np
import pandas as pd
import os
import alex.evaluation as ev  # Assumes plot_multiple_columns is defined here

# === Material Parameters ===
mu0 = 1.0       # Initial shear modulus
lam0 = 1.0      # Initial Lamé parameter
sigma_y = 1.0   # Yield stress
r = 10          # Smoothness parameter
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
    K = lam0 + 2.0 * mu0 / 3.0
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

    for eps0_val in eps0_vals:
        eps_tensor = strain_tensor(eps0_val)
        e_dev = deviatoric(eps_tensor)
        e_eq = np.sqrt(2/3 * np.sum(e_dev**2))
        mu_val = mu_eps(e_eq, mu0, b, sigma_y, r)
        sig_tensor = stress_tensor(eps_tensor, mu_eps, b, sigma_y, r)
        s_dev = deviatoric(sig_tensor)

        sig11_list.append(sig_tensor[0, 0])
        e0_list.append(eps0_val)
        s_dev_norm_list.append(tensor_norm(s_dev))
        e_dev_norm_list.append(tensor_norm(e_dev))

    df1 = pd.DataFrame({0: e0_list, 1: sig11_list})
    df2 = pd.DataFrame({0: e_dev_norm_list, 1: s_dev_norm_list})
    data_sig11.append(df1)
    data_devnorm.append(df2)
    legend_labels.append(fr"$b={b}$")

# === Full plasticity implementation ===
e_p_n = deviatoric(strain_tensor(0))
alpha_n = 0
b_ref = b_values[0]
H = b_ref * mu0 / (1 - b_ref)
sigma_11_plasti = []
e_dev_norm_plasti = []
s_dev_norm_plasti = []

for eps0_val in eps0_vals:
    eps_np1 = strain_tensor(eps0_val)
    s_trial = 2.0 * mu0 * deviatoric(eps_np1 - e_p_n)
    norm_s_trial = np.linalg.norm(s_trial)

    f_tr = norm_s_trial - np.sqrt(2/3)*(sigma_y + H * alpha_n)
    K = lam0 + 2.0 * mu0 / 3.0

    if f_tr <= 0.0:
        e_p_np1 = e_p_n
        alpha_np1 = alpha_n
        s_corrected = s_trial
    else:
        N = s_trial / norm_s_trial
        dgamma = f_tr / (2.0 * mu0 + 2/3 * H)
        e_p_np1 = e_p_n + dgamma * N
        alpha_np1 = alpha_n + np.sqrt(2/3) * dgamma
        eps_e = eps_np1 - e_p_np1
        s_corrected = 2.0 * mu0 * deviatoric(eps_e)

    sigma_np1 = K * np.trace(eps_np1 - e_p_np1) * np.eye(3) + s_corrected

    sigma_11_plasti.append(sigma_np1[0, 0])
    e_dev_norm_plasti.append(tensor_norm(deviatoric(eps_np1)))
    s_dev_norm_plasti.append(tensor_norm(s_corrected))

    e_p_n = e_p_np1
    alpha_n = alpha_np1

# === Append plasticity result to plots ===
df_plastic_sig11 = pd.DataFrame({0: eps0_vals, 1: sigma_11_plasti})
df_plastic_devnorm = pd.DataFrame({0: e_dev_norm_plasti, 1: s_dev_norm_plasti})
data_sig11.append(df_plastic_sig11)
data_devnorm.append(df_plastic_devnorm)
legend_labels.append(fr"Plasticity ($H={H:.2f}\mu_0$)")

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
    ylabel=r"$\sigma_{xx}$ / $\sigma_y$",
    hlines=[[sigma_y]],
    legend_labels=legend_labels,
    x_range=[0, 1.0],
    y_range=[0, 2.5],
    use_colors=True,
    usetex=True,
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
    legend_labels=legend_labels,
    x_range=[0, 1.0],
    y_range=[0, 2.5],
    use_colors=True,
    usetex=True,
    marker_size=4,
)

