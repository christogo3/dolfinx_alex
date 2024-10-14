#!/bin/bash

# Define the list of WSTEG values you want to iterate over
WSTEG_VALUES=(0.05 0.1 0.2 0.4 0.8 1.0 10.0)

# Other parameters to be kept constant
L0=6.0
HSTEG=0.5
VOL_MAT=0.5
MESH_FILE="mesh_holes.xdmf"
LAM_PARAM=1.0
MUE_PARAM=1.0
GC_PARAM=1.0
EPS_FACTOR_PARAM=0.05
ELEMENT_ORDER=1

# Loop over each WSTEG value
for WSTEG in "${WSTEG_VALUES[@]}"; do
  # Calculate WHOLE with awk for floating-point arithmetic
  WHOLE=$(awk "BEGIN {print $WSTEG * (1.0 - $VOL_MAT) / $VOL_MAT}")
  
  # Check if WHOLE calculation succeeded
  if [ -z "$WHOLE" ]; then
      echo "Error: WHOLE calculation failed for WSTEG=$WSTEG"
      exit 1
  fi
  
  # Debug: Print the values of WSTEG and WHOLE
  echo "Debug: WSTEG=$WSTEG, WHOLE=$WHOLE"
  
  # Call the original script with the current WSTEG value
  echo "Running script with WSTEG=$WSTEG and WHOLE=$WHOLE"
  ./script.sh "$L0" "$WSTEG" "$HSTEG" "$WHOLE" "$MESH_FILE" "$LAM_PARAM" "$MUE_PARAM" "$GC_PARAM" "$EPS_FACTOR_PARAM" "$ELEMENT_ORDER"
  
  # Check if the script executed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Script failed for WSTEG=$WSTEG"
    exit 1
  fi
  
  echo "Completed run for WSTEG=$WSTEG"
done

echo "All runs completed successfully."





