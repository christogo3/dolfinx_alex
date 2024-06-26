#!/bin/bash

# Base directory where the simulation folders are located
BASE_DIR="/home/as12vapa/dolfinx_alex/shared/scripts/11-foam-parameter-study"

# Iterate over each simulation folder
for folder_path in "${BASE_DIR}/simulation_*"; do
    if [ -d "${folder_path}" ]; then
        # Path to the job script
        job_script_path="${folder_path}/job_script.sh"

        # Check if the job script exists
        if [ -f "${job_script_path}" ]; then
            # Submit the job script using sbatch
            sbatch "${job_script_path}"
            echo "Submitted job script: ${job_script_path}"
        else
            echo "Job script not found: ${job_script_path}"
        fi
    fi
done
