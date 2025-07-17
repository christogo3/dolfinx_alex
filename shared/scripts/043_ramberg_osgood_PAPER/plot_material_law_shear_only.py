import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Parameters for new G(γ) model ----
#b = 0.1             # stiffness ratio at large strains
gamma_0 = 0.02      # characteristic shear strain
sigma_0 = 80000.0   # MPa, reference stress
#r = 100             # controls transition sharpness


b = 0.001        # Strain hardening parameter
r = 100          # Bauschinger effect parameter
# ---- New shear modulus function (explicit) ----
def compute_G(gamma, b, gamma_0, sigma_0, r):
    term = (1 + abs(gamma / gamma_0)**r)**(1/r)
    G = (b + (1 - b) / term) * sigma_0 / gamma_0
    return G

# ---- Compute shear stress: sigma = G(gamma) * gamma ----
def compute_shear_stress(gamma_vals):
    sigma_vals = []
    for gamma in gamma_vals:
        G = compute_G(gamma, b, gamma_0, sigma_0, r)
        sigma = G * gamma
        sigma_vals.append(sigma)
    return np.array(sigma_vals)

# ---- Shear strain range ----
gamma_vals = np.linspace(-2 * gamma_0, 2 * gamma_0, 1000)
sigma_vals = compute_shear_stress(gamma_vals)

# ---- Plot and Save ----
script_path = os.path.dirname(__file__)
outpath = os.path.join(script_path, "shear_material_law.png")

plt.figure(figsize=(8, 6))
plt.plot(gamma_vals, sigma_vals, label=r'$\sigma = G(\gamma) \cdot \gamma$')
plt.title("Shear Stress-Strain Curve with Nonlinear Shear Modulus")
plt.xlabel(r'Shear Strain $\gamma$')
plt.ylabel(r'Shear Stress $\sigma$ (MPa)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Plot saved to '{outpath}'")


