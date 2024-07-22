#!/bin/bash

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
BASE_DIR="${HPC_SCRATCH}/${working_dir}"

# Define the directory where the job template is located
JOB_TEMPLATE_DIR="./00_jobs"
JOB_TEMPLATE_PATH="${JOB_TEMPLATE_DIR}/job_template.sh"

# Function to extract parameters from folder name
extract_parameters() {
    local folder_name=$1
    local name_only=${folder_name#"simulation_"}  # Remove "simulation_" prefix
    local elements=($(echo "${name_only//_/ }"))  # Split by underscores

    local mesh_file="${elements[2]}_${elements[3]}"  # Combine 3rd and 4th elements with underscore
    local lam_param=${elements[4]#lam}
    local mue_param=${elements[5]#mue}
    local gc_param=${elements[6]#Gc}
    local eps_factor_param=${elements[7]#eps}
    local element_order=${elements[8]#order}

    echo "${mesh_file} ${lam_param} ${mue_param} ${gc_param} ${eps_factor_param} ${element_order}"
}

# Function to generate a job script for a given simulation folder
generate_job_script() {
    local folder_name=$1
    local mesh_file=$2
    local lam_param=$3
    local mue_param=$4
    local gc_param=$5
    local eps_factor_param=$6
    local element_order=$7

    # Set job_name to only the mesh type name
    local job_name="${mesh_file}"

    # Read the template and replace placeholders
    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{MESH_FILE}|${mesh_file}|g" \
        -e "s|{LAM_PARAM}|${lam_param}|g" \
        -e "s|{MUE_PARAM}|${mue_param}|g" \
        -e "s|{GC_PARAM}|${gc_param}|g" \
        -e "s|{EPS_FACTOR_PARAM}|${eps_factor_param}|g" \
        -e "s|{ELEMENT_ORDER}|${element_order}|g" \
        "${JOB_TEMPLATE_PATH}" > "${BASE_DIR}/${folder_name}/job_script.sh"
}











