#!/bin/bash

# Define the list of WSTEG values you want to iterate over
#WSTEG_VALUES=(0.05 0.1 0.2 0.4 0.5 1.0)

#WSTEG_VALUES=(0.05 0.075 0.1 0.125 0.15 0.2)
WSTEG_VALUES=(4.0)

# Other parameters to be kept constant
MESH_FILE="mesh_fracture_adaptive.xdmf"
NHOLES=4
DHOLE=1.0
E0=0.02
E1=0.7
EPS_PARAM=$(awk "BEGIN {print 5 * $E0}")
# E0=0.01
# EPS_PARAM=$(awk "BEGIN {print 10 * $E0}") # temporary for small steg width
LAM_MICRO_PARAM=1.0
MUE_MICRO_PARAM=1.0
GC__MICRO_PARAM=1.0
ELEMENT_ORDER=1

# Loop over each WSTEG value
for WSTEG in "${WSTEG_VALUES[@]}"; do

  # Call the original script with the current WSTEG value
  echo "Running script with WSTEG=$WSTEG"
  ./script_adaptive.sh "$NHOLES" "$WSTEG" "$DHOLE" "$E0" "$E1" "$MESH_FILE" "$LAM_MICRO_PARAM" "$MUE_MICRO_PARAM" "$GC_MICRO_PARAM" "$EPS_PARAM" "$ELEMENT_ORDER"
  
  # Check if the script executed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Script failed for WSTEG=$WSTEG"
    exit 1
  fi
  
  echo "Completed run for WSTEG=$WSTEG"
done

echo "All runs completed successfully."





