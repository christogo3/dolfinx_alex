#!/bin/bash

# Define an array of script paths to execute
SCRIPTS=(
  "./master_e_gc.sh"
  "./master_height.sh"
  "./master_width.sh"
  "./master_gc.sh"
  "./master_e.sh"
  "./master_nu.sh"
  "./master_eps.sh"
)

# Execute each script in the array
for SCRIPT in "${SCRIPTS[@]}"; do
  echo "Executing $SCRIPT..."
  
  # Check if the script is executable
  if [[ -x "$SCRIPT" ]]; then
    # Run the script
    "$SCRIPT"
    
    # Check the exit status
    if [[ $? -ne 0 ]]; then
      echo "Error: $SCRIPT failed to execute successfully. Exiting."
      exit 1
    else
      echo "$SCRIPT completed successfully."
    fi
  else
    echo "Error: $SCRIPT is not executable or does not exist. Exiting."
    exit 1
  fi
done

echo "All scripts executed successfully!"



