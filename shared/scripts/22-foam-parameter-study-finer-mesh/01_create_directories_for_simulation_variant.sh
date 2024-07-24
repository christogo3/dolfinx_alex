#!/bin/bash

# global length scale
L=1.2720814740168862
# Define the h_coarse_mean
h_coarse_mean=0.024636717648428213
# Define the pore_size_coarse
pore_size_coarse=0.183

epsilon_to_h_min_ratio=2.0
pore_size_to_eps_min_ratio=2.0

epsilon_min=epsilon_to_h_min_ratio*h_coarse_mean
epsilon_max=pore_size_coarse/pore_size_to_eps_min_ratio

inv_epsilon_max=1.0/epsilon_min
inv_epsilon_min=1.0/epsilon_max

number_of_eps_values_to_compute=4

# Compute EPS_FACTOR_PARAMS
EPS_FACTOR_PARAMS=()
step=$(echo "($inv_epsilon_max - $inv_epsilon_min) / ($number_of_eps_values_to_compute - 1)" | bc -l)

for ((i=0; i<number_of_eps_values_to_compute; i++)); do
    value=$(echo "$inv_epsilon_min + $i * $step" | bc -l)
    EPS_FACTOR_PARAMS+=($value)
done

# Print EPS_FACTOR_PARAMS for verification
echo "EPS_FACTOR_PARAMS: ${EPS_FACTOR_PARAMS[@]}"

# Initialize the h_all associative array
# declare -A h_all
# h_all=(
#     ["coarse_pores"]=$h_coarse_mean
#     ["medium_pores"]=$(echo "$h_coarse_mean / 2.0" | bc -l)
#     ["fine_pores"]=$(echo "$h_coarse_mean / 4.0" | bc -l)
# )

# # Initialize the pore_size_all associative array
# declare -A pore_size_all
# pore_size_all=(
#     ["coarse_pores"]=$pore_size_coarse
#     ["medium_pores"]=$(echo "$pore_size_coarse / 2.0" | bc -l)
#     ["fine_pores"]=$(echo "$pore_size_coarse / 4.0" | bc -l)
# )

# Define the possible values for each parameter
MESH_FILES=("coarse_pores" "medium_pores")
LAM_MUE_PAIRS=(
    "1.0 1.0" 
    "1.5 1.0" 
    "1.0 1.5"
    "10.0 10.0" 
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
    folder_name="simulation_${current_time}_${mesh_file}_lam${lam_param}_mue${mue_param}_Gc${gc_param}_eps${eps_factor_param}_order${element_order}"

    # Create the new directory
    mkdir -p "${BASE_WORKING_DIR}/${folder_name}"
    
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
                
                # Set element order to 1
                element_order=1
                replicate_folder "$mesh_file" "$lam_param" "$mue_param" "$gc_param" "$scaled_eps_factor_param" "$element_order"
            done
        done
    done
done







