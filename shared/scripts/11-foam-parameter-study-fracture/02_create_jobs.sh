#!/bin/bash

# Define the base directory where the simulation folders are located
BASE_DIR="/home/as12vapa/dolfinx_alex/shared/scripts/11-foam-parameter-study-fracture"

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

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")

        # Extract parameters from folder name
        params=$(extract_parameters "${folder_name}")
        set -- $params  # set positional parameters

        # Call generate_job_script with extracted parameters
        generate_job_script "${folder_name}" "$1" "$2" "$3" "$4" "$5" "$6"
    fi
done










