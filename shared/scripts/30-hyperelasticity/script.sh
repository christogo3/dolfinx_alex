#!/bin/bash

# Bash script to run Python simulation script with specified parameter pairs

# Define pairs of (lam, mu)
lam_mu_pairs=(
    "1.5 1.0"
    "1.0 1.0"
    "0.667 1.0"
    "1.0 0.429"
    "0.25 1.0"
    "0.0 1.0"
)

# Define values for rho0
rho0_values=(1.0)

# Define values for psi
psi_values=("SV" "NH")

# Python script to execute
#PYTHON_SCRIPT="hyperelastic_square_tension.py"
PYTHON_SCRIPT="hyperelastic_square_shear.py"

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


python3 "$PYTHON_SCRIPT" --lam "-0.167" --mue "1.0" --rh0 "1.0" --psi "NH"
#echo "Simulation completed for lam=$lam, mu=$mu, rho0=$rho0, psi=$psi"

echo "All simulations completed."

