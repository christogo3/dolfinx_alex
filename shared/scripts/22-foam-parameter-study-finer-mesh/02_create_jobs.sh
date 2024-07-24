#!/bin/bash

# Define the base directory where the simulation folders are located
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

# Function to determine memory, processor, and time settings based on mesh_file
get_memory_processors_and_time() {
    local mesh_file=$1
    local memory_value
    local processor_number
    local time

    # Define memory, processor, and time settings based on mesh_file (example logic)
    case "$mesh_file" in
        "coarse_pores")
            memory_value=4000
            processor_number=64
            time="6000"
            ;;
        "medium_pores")
            memory_value=6000
            processor_number=64
            time="10080"
            ;;
        "fine_pores")
            memory_value=9000
            processor_number=192
            time="10080"
            ;;
        *)
            memory_value=6000
            processor_number=96
            time="10080"
            ;;
    esac

    echo "${memory_value} ${processor_number} ${time}"
}

# Function to generate a job script for a given simulation folder
generate_job_script() {
    local folder_name=$1
    local job_name=$2
    local mesh_file=$3
    local lam_param=$4
    local mue_param=$5
    local gc_param=$6
    local eps_factor_param=$7
    local element_order=$8
    local memory_value=$9
    local processor_number=${10}
    local time=${11}

    # Read the template and replace placeholders
    sed -e "s|{FOLDER_NAME}|${folder_name}|g" \
        -e "s|{JOB_NAME}|${job_name}|g" \
        -e "s|{MESH_FILE}|${mesh_file}|g" \
        -e "s|{LAM_PARAM}|${lam_param}|g" \
        -e "s|{MUE_PARAM}|${mue_param}|g" \
        -e "s|{GC_PARAM}|${gc_param}|g" \
        -e "s|{EPS_FACTOR_PARAM}|${eps_factor_param}|g" \
        -e "s|{ELEMENT_ORDER}|${element_order}|g" \
        -e "s|{MEMORY_VALUE}|${memory_value}|g" \
        -e "s|{PROCESSOR_NUMBER}|${processor_number}|g" \
        -e "s|{TIME}|${time}|g" \
        "${JOB_TEMPLATE_PATH}" > "${BASE_DIR}/${folder_name}/job_script.sh"
}

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")

        # Extract parameters from folder name
        params=$(extract_parameters "${folder_name}")
        set -- $params  # set positional parameters
        job_name="$1"  # Set job_name to mesh_file

        # Debugging: Print extracted parameters
        echo "Extracted parameters: $params"

        # Get memory, processor, and time settings based on mesh_file
        mem_proc_time=$(get_memory_processors_and_time "$1")
        set -- $mem_proc_time  # set positional parameters for memory, processor, and time settings
        memory_value=$1
        processor_number=$2
        time=$3

        # Debugging: Print memory, processor, and time settings
        echo "Memory: $memory_value, Processors: $processor_number, Time: $time"

        # Call generate_job_script with extracted parameters and memory/processor/time settings
        generate_job_script "${folder_name}" "${job_name}" "$1" "$2" "$3" "$4" "$5" "$6" "${memory_value}" "${processor_number}" "${time}"
    fi
done

















