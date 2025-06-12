import numpy as np
import matplotlib.pyplot as plt
import os

# === User-defined parameters ===
E = 2.5  # Young's modulus in Pascals
param_pairs = [
    (0.01, 1.0),
    (0.01, 2.0),
    (0.01, 4.0),
    (0.01, 8.0)
]  # List of (C, n) pairs

# === Strain range (epsilon) ===
epsilon = np.linspace(1e-6, 0.02, 100)  # avoid zero to prevent division issues

# === Plotting ===
plt.figure(figsize=(10, 7))

for C, n in param_pairs:
    sigma = E * (1 / (C ** (1 / n))) * (epsilon ** (1 / n))
    plt.plot(epsilon, sigma, label=f'C={C}, n={n}')

plt.xlabel('Strain (ε)')
plt.ylabel('Stress (σ) [Pa]')
plt.title('Stress-Strain Relation: σ = E * (1 / C^(1/n)) * ε^(1/n)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save to file ===
script_path = os.path.dirname(__file__)
output_file = os.path.join(script_path, 'ramberg-osgood-diewald.png')
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
