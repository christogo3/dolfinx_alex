#!/bin/bash

# Paths to subfolders
SUBFOLDER1="032-crack-nucleation-2D-PAPER"
SUBFOLDER2="033-crack-nucleation-diamond-PAPER"

# Script names
SCRIPT1="master_all.sh"
SCRIPT2="master_all.sh"

# Function to navigate into a folder, execute the script, and return
execute_script() {
    local folder=$1
    local script=$2
    
    if [[ -d "$folder" ]]; then
        cd "$folder" || { echo "Error: Could not enter $folder"; return 1; }
        
        if [[ -x "$script" ]]; then
            echo "Executing $script in $folder..."
            ./"$script"
        else
            echo "Error: $script not found or not executable in $folder."
        fi
        
        # Return to the original directory
        cd - > /dev/null || { echo "Error: Could not return to the original directory"; return 1; }
    else
        echo "Error: Directory $folder does not exist."
    fi
}

# Execute the scripts in their respective subfolders
execute_script "$SUBFOLDER1" "$SCRIPT1"
execute_script "$SUBFOLDER2" "$SCRIPT2"
