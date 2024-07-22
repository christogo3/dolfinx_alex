#!/bin/bash

# Usage: ./remove_folders.sh /path/to/directory substring1 substring2 substring3

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

        # Flag to determine if the folder should be removed
        REMOVE_FOLDER=false

        # Look for *.err.id files in the folder
        for ERR_FILE in "$FOLDER"/*.err.*; do
            if [[ -f "$ERR_FILE" && "$ERR_FILE" == *.err.* ]]; then
                # Check if the *.err.id file is not empty
                if [ -s "$ERR_FILE" ]; then
                    REMOVE_FOLDER=true
                    break
                fi
            fi
        done

        # If folder name contains any of the substrings
        if [ "$CONTAINS_SUBSTRING" = true ]; then
            REMOVE_FOLDER=true
        fi

        # Remove the folder if it meets the criteria
        if [ "$REMOVE_FOLDER" = true ]; then
            echo "Removing folder: $FOLDER"
            rm -rf "$FOLDER"
        fi
    fi
done
