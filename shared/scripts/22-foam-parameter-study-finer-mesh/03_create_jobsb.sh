#!/bin/bash

# Define the base directory where the job scripts are located
BASE_DIR="${HPC_SCRATCH}/22-foam-parameter-study-finer-mesh"

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
            processor_number=96
            time="10080"
            ;;
        "fine_pores")
            memory_value=8000
            processor_number=128
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

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")

        # Extract parameters from folder name
        IFS='_' read -r -a elements <<< "${folder_name#simulation_}"
        mesh_file="${elements[2]}_${elements[3]}"

        # Get memory, processor, and time settings based on mesh_file
        mem_proc_time=$(get_memory_processors_and_time "$mesh_file")
        set -- $mem_proc_time  # set positional parameters for memory, processor, and time settings
        memory_value=$1
        processor_number=$2
        time=$3

        # Define the path to the job script to be updated
        job_script_path="${folder_path}/job_script.sh"

        # Update the job script with correct values
        sed -i -e "s|{MEMORY_VALUE}|${memory_value}|g" \
               -e "s|{PROCESSOR_NUMBER}|${processor_number}|g" \
               -e "s|{TIME}|${time}|g" \
               "${job_script_path}"

        # Print the updated job script for debugging
        echo "Updated job script for folder ${folder_name}:"
        cat "${job_script_path}"
    fi
done

















