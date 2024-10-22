#!/bin/bash

# global length scale
L=1.2720814740168862

# Define the h_coarse_mean
h_coarse_mean=0.024636717648428213

# Define the pore_size_coarse
pore_size_coarse=0.183

# Constants for calculation
epsilon_to_h_min_ratio=2.0
pore_size_to_eps_min_ratio=2.0

# Compute epsilon_min and epsilon_max
epsilon_min=$(echo "$epsilon_to_h_min_ratio * $h_coarse_mean" | bc -l)
epsilon_max=$(echo "$pore_size_coarse / $pore_size_to_eps_min_ratio" | bc -l)

# Compute inv_epsilon_max and inv_epsilon_min
inv_epsilon_max=$(echo "1.0 / $epsilon_min * $L" | bc -l)
inv_epsilon_min=$(echo "1.0 / $epsilon_max * $L" | bc -l)

# Number of EPS values to compute
number_of_eps_values_to_compute=4

# Compute step value
step=$(echo "($inv_epsilon_max - $inv_epsilon_min) / ($number_of_eps_values_to_compute - 1)" | bc -l)

# Function to format floating-point numbers
format_number() {
    printf "%.4f" "$1"
}

# Compute EPS_FACTOR_PARAMS with formatted output
EPS_FACTOR_PARAMS=()
for ((i=0; i<number_of_eps_values_to_compute; i++)); do
    value=$(echo "$inv_epsilon_min + $i * $step" | bc -l)
    formatted_value=$(format_number "$value")
    EPS_FACTOR_PARAMS+=($formatted_value)
done

# Print EPS_FACTOR_PARAMS for verification
echo "EPS_FACTOR_PARAMS: ${EPS_FACTOR_PARAMS[@]}"

# Define the possible values for each parameter
MESH_FILES=("coarse_pores")
LAM_MUE_PAIRS=(
    "1.0    1.0"
    "1.5    1.0"
    "0.6667 1.0" 
)
GC_FACTORS=(25.0 50.0)  # Factors to compute Gc

# Define the template folder
TEMPLATE_FOLDER="000_template"

# Get the current directory of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Get the name of the folder in which the bash script is located
working_dir=$(basename "$SCRIPT_DIR")

# Ensure HPC_SCRATCH is defined
if [ -z "$HPC_SCRATCH" ]; then
    echo "Error: HPC_SCRATCH is not defined."
    exit 1
fi

# Create the base working directory if it doesn't exist
BASE_WORKING_DIR="${HPC_SCRATCH}/${working_dir}"
mkdir -p "$BASE_WORKING_DIR"

# Function to replicate the template folder for each combination of parameters
replicate_folder() {
    local mesh_file=$1
    local lam_param=$2
    local mue_param=$3
    local gc_param=$4
    local eps_factor_param=$5
    local element_order=$6

    # Create a unique folder name
    current_time=$(date +%Y%m%d_%H%M%S)
    folder_name="simulation_${current_time}_${mesh_file}_lam$(format_number "$lam_param")_mue$(format_number "$mue_param")_Gc$(format_number "$gc_param")_eps$(format_number "$eps_factor_param")_order${element_order}"

    # Create the new directory
    mkdir -p "${BASE_WORKING_DIR}/${folder_name}"
    
    # Create the scratch/as12vapa directory inside the new simulation folder
    #mkdir -p "${BASE_WORKING_DIR}/${folder_name}/scratch/as12vapa"

    # Copy the contents of the template folder to the new directory
    rsync -av --exclude='000_template' "${SCRIPT_DIR}/${TEMPLATE_FOLDER}/" "${BASE_WORKING_DIR}/${folder_name}/"
}

# Iterate over all combinations of parameters
for mesh_file in "${MESH_FILES[@]}"; do
    for lam_mue_pair in "${LAM_MUE_PAIRS[@]}"; do
        lam_param=$(echo $lam_mue_pair | cut -d ' ' -f 1)
        mue_param=$(echo $lam_mue_pair | cut -d ' ' -f 2)
        for factor in "${GC_FACTORS[@]}"; do
            for eps_factor_param in "${EPS_FACTOR_PARAMS[@]}"; do
                # Scale EPS_FACTOR_PARAMS based on mesh_file
                if [ "$mesh_file" == "medium_pores" ]; then
                    scaled_eps_factor_param=$(echo "$eps_factor_param * 2" | bc -l)
                elif [ "$mesh_file" == "fine_pores" ]; then
                    scaled_eps_factor_param=$(echo "$eps_factor_param * 4" | bc -l)
                else
                    scaled_eps_factor_param=$eps_factor_param
                fi
                
                # Compute gc_param
                gc_param=$(echo "$factor / $scaled_eps_factor_param" | bc -l)
                
                # Format gc_param and scaled_eps_factor_param
                formatted_gc_param=$(format_number "$gc_param")
                formatted_scaled_eps_factor_param=$(format_number "$scaled_eps_factor_param")
                
                # Set element order to 1
                element_order=1
                replicate_folder "$mesh_file" "$lam_param" "$mue_param" "$formatted_gc_param" "$formatted_scaled_eps_factor_param" "$element_order"
            done
        done
    done
done









