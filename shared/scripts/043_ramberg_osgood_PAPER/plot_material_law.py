# alternative model from https://en.wikipedia.org/wiki/Ramberg%E2%80%93Osgood_relationship


import numpy as np
import matplotlib.pyplot as plt
import os

script_path = os.path.dirname(__file__)

# Define the normalized stress-strain function
def normalized_stress_strain(eps_bar, b, r):
    return b * eps_bar + ((1 - b) * eps_bar) / ((1 + np.abs(eps_bar)**r)**(1/r))

def G(eps_bar, b, r):
    return b  + ((1 - b)) / ((1 + np.abs(eps_bar)**r)**(1/r))

# Parameters
sigma_0 = 400  # Yield stress in MPa, for example
eps_0 = 0.02   # Yield strain, for example
b = 0.001        # Strain hardening parameter
r = 100          # Bauschinger effect parameter

# Strain range in physical units
eps = np.linspace(-2 * eps_0, 2 * eps_0, 500)
eps_bar = eps / eps_0
#sigma_bar = normalized_stress_strain(eps_bar, b, r)
sigma_bar = G(eps_bar, b, r) * eps_bar
sigma = sigma_bar * sigma_0

# Output path
outpath = os.path.join(script_path, "material_laws.png")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(eps, sigma, label=r'$\sigma(\varepsilon)$')
plt.title('Stress-Strain Curve')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\sigma$ (MPa)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outpath, dpi=300)
plt.close()

print(f"Plot saved to '{outpath}'")

