#!/bin/bash

# Define the possible values for each parameter
# MESH_FILES=("coarse_pores", "medium_pores", "fine_pores")
# LAM_MUE_PAIRS=(
#     "1.0 1.0" 
#     "1.5 1.0" 
#     "1.0 1.5"
#     "10.0 10.0" 
# )
# GC_PARAMS=(0.5 0.75 1.0 1.25 1.5)
# EPS_FACTOR_PARAMS=(25.0 33.0 40.0 50.0 100.0)

# Define the possible values for each parameter
MESH_FILES=("fine_pores")
LAM_MUE_PAIRS=(
    "1.0 1.0" 
)
GC_PARAMS=(1.0)
EPS_FACTOR_PARAMS=(50.0)

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
        for gc_param in "${GC_PARAMS[@]}"; do
            for eps_factor_param in "${EPS_FACTOR_PARAMS[@]}"; do
                # Set element order to 1
                element_order=1
                replicate_folder "$mesh_file" "$lam_param" "$mue_param" "$gc_param" "$eps_factor_param" "$element_order"
            done
        done
    done
done




