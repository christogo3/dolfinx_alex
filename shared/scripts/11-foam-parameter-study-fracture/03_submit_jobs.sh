#!/bin/bash

# Base directory where the simulation folders are located
BASE_DIR="/home/as12vapa/dolfinx_alex/shared/scripts/11-foam-parameter-study"

echo "Starting job submission script"
echo "Base directory: ${BASE_DIR}"

# Iterate over each simulation folder
for folder_path in ${BASE_DIR}/simulation_*; do
    echo "Processing folder: ${folder_path}"
    
    if [ -d "${folder_path}" ]; then
        # Path to the job script
        job_script_path="${folder_path}/job_script.sh"
        
        echo "Looking for job script: ${job_script_path}"

        # Check if the job script exists
        if [ -f "${job_script_path}" ]; then
            echo "Job script found: ${job_script_path}"
            
            # Submit the job script using sbatch
            sbatch "${job_script_path}"
            
            if [ $? -eq 0 ]; then
                echo "Successfully submitted job script: ${job_script_path}"
            else
                echo "Failed to submit job script: ${job_script_path}"
            fi
        else
            echo "Job script not found: ${job_script_path}"
        fi
    else
        echo "Not a directory: ${folder_path}"
    fi
done

echo "Job submission script completed"
