#!/bin/bash

# Define specific values for WSTEG (array of values to vary)
WSTEG_VALUES=(1.0)  # Example WSTEG values

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

# Function to replicate the template folder for each WSTEG value
replicate_folder() {
    local wsteg_value=$1

    # Create a unique folder name that includes WSTEG value
    current_time=$(date +%Y%m%d_%H%M%S)
    folder_name="simulation_${current_time}_WSTEG${wsteg_value}"

    # Create the new directory
    mkdir -p "${BASE_WORKING_DIR}/${folder_name}"
    
    # Create the scratch/as12vapa directory inside the new simulation folder
    #mkdir -p "${BASE_WORKING_DIR}/${folder_name}/scratch/as12vapa"

    # Copy the contents of the template folder to the new directory
    rsync -av --exclude='000_template' "${SCRIPT_DIR}/${TEMPLATE_FOLDER}/" "${BASE_WORKING_DIR}/${folder_name}/"
}

# Iterate over all WSTEG values
for wsteg_value in "${WSTEG_VALUES[@]}"; do
    replicate_folder "$wsteg_value"
done









