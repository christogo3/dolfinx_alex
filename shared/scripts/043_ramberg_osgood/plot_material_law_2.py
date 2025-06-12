# alternative model from https://en.wikipedia.org/wiki/Ramberg%E2%80%93Osgood_relationship


import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Parameters ----
E_0 = 2.5      # MPa, initial Young's modulus
nu_0 = 0.25 #25    # Initial Poisson's ratio
eps_0 = 0.2    # 1D yield strain

# ---- Nonlinear shear model parameters ----
b = 0.1     # Strain hardening parameter
r = 10.0      # Transition sharpness

# ---- Derived constants ----
K_0 = E_0 / (3 * (1 - 2 * nu_0))         # Bulk modulus
G_0 = E_0 / (2 * (1 + nu_0))             # Initial shear modulus
#eps_shear_0 = (2 / 3) * eps_0                # Yield shear strain
#tau_0 = G_0 * 2.0*eps_shear_0                    # Reference shear stress


eps_lateral_crit = -nu_0 * eps_0
eps_crit = np.array([
            [eps_0, 0, 0],
            [0, eps_lateral_crit, 0],
            [0, 0, eps_lateral_crit]
        ], dtype=float)
tr_eps_crit = np.trace(eps_crit)
identity = np.eye(3)
eps_crit_dev = eps_crit - (tr_eps_crit / 3.0) * identity
norm_eps_crit_dev = np.linalg.norm(eps_crit_dev, 'fro')

norm_sig_dev_crit = G_0*2.0*norm_eps_crit_dev
# ---- Equivalent shear strain for 1D loading ----
# def compute_eps_shear_1D_stress(eps_11, nu):
#     eps_22 = -nu * eps_11
#     eps_33 = eps_22
#     eps_shear = (2/3) * np.sqrt(
#         (eps_11 - eps_22)**2 +
#         (eps_22 - eps_33)**2 +
#         (eps_33 - eps_11)**2
#     )
#     return eps_shear

# ---- Explicit shear modulus model ----
# def compute_G(eps_shear, b, eps_shear_0, r):
#     term = (1 + abs(eps_shear / (eps_shear_0))**r)**(1/r)
#     G = (b + (1 - b) / term) * (tau_0 / (eps_shear_0))
#     return G


def compute_G(norm_eps_dev, b, r,norm_eps_crit_dev,norm_sig_dev_crit):
    return (b  + ((1 - b)) / ((1 + np.abs(norm_eps_dev/norm_eps_crit_dev)**r)**(1/r))) * ( norm_sig_dev_crit / (norm_eps_crit_dev*2.0)  )

# ---- Compute σ = E(γ) * ε with updated Poisson's ratio ----
def compute_sigma(eps_vals):
    sigma_vals = []
    sigma_dev_vals = []
    eps_dev_vals = []
    
    for eps_11 in eps_vals:
        # Guess ν using initial G for eps_shear computation
        #nu_guess = nu_0
        #eps_shear = compute_eps_shear_1D_stress(eps_11, nu_guess)
        
        eps_lateral = -nu_0 * eps_11
        eps = np.array([
            [eps_11, 0, 0],
            [0, eps_lateral, 0],
            [0, 0, eps_lateral]
        ], dtype=float)
        
        tr_eps = np.trace(eps)
        
        identity = np.eye(3)
        eps_dev = eps - (tr_eps / 3.0) * identity
        norm_eps_dev = np.linalg.norm(eps_dev, 'fro')
        
        # Get nonlinear shear modulus
        G = compute_G(norm_eps_dev, b, r,norm_eps_crit_dev,norm_sig_dev_crit)
        
        # Update ν with new G
        nu = (3 * K_0 - 2 * G) / (2 * (3 * K_0 + G))
        
        # Recalculate lateral strains with updated ν
        # eps_22 = -nu_0 * eps_11
        # eps_33 = eps_22
        # tr_eps = eps_11 + eps_22 + eps_33
        
        # Deviatoric strain component (only 11 since uniaxial)
        eps_22 = eps_lateral
        eps_33 = eps_lateral
        
        eps_dev_11 = eps_11 - tr_eps / 3.0
        eps_dev_22 = eps_22 - tr_eps / 3.0
        eps_dev_33 = eps_33 - tr_eps / 3.0
        
        eps_dev_vals.append(eps_dev_11)
        
        # Total axial stress
        la = K_0 - ((2.0*G) / 3.0)
        
        sigma_11 = la * tr_eps + 2 * G * eps_11
        sigma_22 = la * tr_eps + 2 * G * eps_22
        sigma_33 = la * tr_eps + 2 * G * eps_33
        tr_sigma = sigma_11 + sigma_22 +sigma_33
        
        sigma_vals.append(sigma_11)
        
        # Deviatoric stress component
        sigma_dev_11 = sigma_11 - (tr_sigma) / 3.0  # only deviatoric part
        sigma_dev_22 = sigma_22 - (tr_sigma) / 3.0
        sigma_dev_33 = sigma_33 - (tr_sigma) / 3.0
        
        norm_dev = np.sqrt(sigma_dev_11**2+sigma_dev_22**2 + sigma_dev_33**2)
        sigma_dev_vals.append(norm_dev) # max shear stress

    return (
        np.array(sigma_vals),
        np.array(sigma_dev_vals),
        np.array(eps_dev_vals)
    )

# ---- Strain range ----
eps_vals = np.linspace(-2 * eps_0, 2 * eps_0, 500)
sig_vals, sig_dev_vals, eps_dev_vals = compute_sigma(eps_vals)

# ---- Plot and Save ----
script_path = os.path.dirname(__file__)
outpath = os.path.join(script_path, "nonlinear_material_law.png")

plt.figure(figsize=(8, 6))

# Main curves
plt.plot(eps_vals, sig_vals, label=r'Total: $\sigma_{11}$ vs $\varepsilon_{11}$')
plt.plot(eps_vals, sig_dev_vals, label=r'Deviatoric: $\sigma_{11}^\prime$ vs $\varepsilon_{11}^\prime$', linestyle='--')

# Auxiliary lines (subtle grey)
aux_color = 'grey'
aux_style = {'color': aux_color, 'linestyle': '--', 'linewidth': 1}

# 1. Initial Young's modulus line
eps_aux = np.linspace(min(eps_vals), max(eps_vals), 500)
plt.plot(eps_aux, E_0 * eps_aux, label='Initial Young\'s Modulus', **aux_style)

# 2. Initial shear modulus line (shear stress vs deviatoric strain)
plt.plot(eps_aux, 2 * G_0 * eps_aux, label='Initial Shear Response', **aux_style)

# 3. Horizontal line at tau_0
plt.axhline(norm_sig_dev_crit, label=r'$\tau_0$', **aux_style)

# 4. Vertical lines at ±eps_shear_0
plt.axvline(eps_0, label=r'$\varepsilon_{s0}$', **aux_style)
plt.axvline(-eps_0, **aux_style)

plt.title("1D Stress-Strain and Deviatoric Response")
plt.xlabel("Strain")
plt.ylabel("Stress (MPa)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Plot saved to '{outpath}'")


