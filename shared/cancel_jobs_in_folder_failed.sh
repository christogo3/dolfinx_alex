#!/bin/bash

# Usage: ./cancel_jobs.sh /path/to/directory substring1 substring2 substring3

# Check if at least 2 arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directory-path> <substring1> [<substring2> ...]"
    exit 1
fi

# Extract the directory path from the first argument
DIRECTORY_PATH=$1
shift  # Shift the arguments so $@ now contains only the substrings

# Create an array from the remaining arguments (substrings)
SUBSTRINGS=("$@")

# Iterate over all folders in the given directory path
for FOLDER in "$DIRECTORY_PATH"/*; do
    if [ -d "$FOLDER" ]; then
        FOLDER_NAME=$(basename "$FOLDER")

        # Check if folder name contains any of the substrings
        CONTAINS_SUBSTRING=false
        for SUBSTRING in "${SUBSTRINGS[@]}"; do
            if [[ "$FOLDER_NAME" == *"$SUBSTRING"* ]]; then
                CONTAINS_SUBSTRING=true
                break
            fi
        done

        # Look for *.err.id files in the folder
        for ERR_FILE in "$FOLDER"/*.err.*; do
            if [[ -f "$ERR_FILE" && "$ERR_FILE" == *.err.* ]]; then
                # Extract the id from the file name
                ID=$(echo "$ERR_FILE" | grep -oP '\d+$')

                # Check if the *.err.id file is not empty or folder name contains any of the substrings
                if [ -s "$ERR_FILE" ] || [ "$CONTAINS_SUBSTRING" = true ]; then
                    echo "Running: scancel $ID"
                    scancel "$ID"
                fi
            fi
        done
    fi
done

