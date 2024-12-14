#!/bin/bash

# Bash script to run Python simulation script with specified parameter pairs

# Define pairs of (lam, mu)
lam_mu_pairs=(
    "0.667 1.0"
)

# Define values for rho0
rho0_values=(1.0)

# Define values for psi
psi_values=("NH")

# Python script to execute
PYTHON_SCRIPT="hyperelastic_bending.py"

# Loop over combinations of parameters
for pair in "${lam_mu_pairs[@]}"; do
    # Extract lam and mu from the pair
    lam=$(echo $pair | awk '{print $1}')
    mu=$(echo $pair | awk '{print $2}')

    for rho0 in "${rho0_values[@]}"; do
        for psi in "${psi_values[@]}"; do
            # Print simulation parameters
            echo "Running simulation with lam=$lam, mu=$mu, rho0=$rho0, psi=$psi"

            # Run the Python script with the current parameters
            python3 "$PYTHON_SCRIPT" --lam "$lam" --mue "$mu" --rh0 "$rho0" --psi "$psi"

            echo "Simulation completed for lam=$lam, mu=$mu, rho0=$rho0, psi=$psi"
        done
    done
done

echo "All simulations completed."

