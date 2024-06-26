#!/bin/bash

# Define the possible values for each parameter
# MESH_FILES=("coarse_pores" "medium_pores" "fine_pores")
# LAM_MUE_PAIRS=(
#     "10.0 20.0"
#     "20.0 30.0"
#     "30.0 40.0"
# )
# GC_PARAMS=(1.0 2.0 3.0)
# EPS_FACTOR_PARAMS=(50.0 100.0 150.0)

MESH_FILES=("coarse_pores")
LAM_MUE_PAIRS=(
    "10.0 20.0"
)
GC_PARAMS=(1.0)
EPS_FACTOR_PARAMS=(50.0 100.0)

# Define the template folder
TEMPLATE_FOLDER="000_template"

# Get the current directory of the script
SCRIPT_DIR=$(dirname "$0")

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

    # Create a new directory based on the template
    cp -r "${SCRIPT_DIR}/${TEMPLATE_FOLDER}" "${SCRIPT_DIR}/${folder_name}"
}

# Iterate over all combinations of parameters
for mesh_file in "${MESH_FILES[@]}"; do
    for lam_mue_pair in "${LAM_MUE_PAIRS[@]}"; do
        lam_param=$(echo $lam_mue_pair | cut -d ' ' -f 1)
        mue_param=$(echo $lam_mue_pair | cut -d ' ' -f 2)
        for gc_param in "${GC_PARAMS[@]}"; do
            for eps_factor_param in "${EPS_FACTOR_PARAMS[@]}"; do
                # Set element order based on mesh_file
                if [ "$mesh_file" == "fine_pores" ]; then
                    element_order=1
                else
                    element_order=2
                fi
                replicate_folder "$mesh_file" "$lam_param" "$mue_param" "$gc_param" "$eps_factor_param" "$element_order"
            done
        done
    done
done

