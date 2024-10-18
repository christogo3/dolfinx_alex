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

# Iterate over each simulation folder in the base directory
for folder_path in "${BASE_DIR}"/simulation_*; do
    if [ -d "${folder_path}" ]; then
        folder_name=$(basename "${folder_path}")

        # Set fixed values for memory, processors, and time
        memory_value=4000
        processor_number=32
        time=1440

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


















