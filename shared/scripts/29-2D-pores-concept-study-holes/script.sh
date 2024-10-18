#!/bin/bash

# Default values for parameters
NHOLES=${1:-3} # needs to be int
WSTEG=${2:-0.1}
DHOLE=${3:-0.25}
E0=${4:-0.01}
MESH_FILE=${5:-"mesh_fracture.xdmf"} # for fracture simulation
LAM_MICRO_PARAM=${6:-1.0}
MUE_MICRO_PARAM=${7:-1.0}
GC_MICRO_PARAM=${8:-1.0}


# Calculate EPS_PARAM as 6 times E0 using awk if not provided by user
EPS_PARAM=$(awk "BEGIN {print 5 * $E0}")
EPS_PARAM=${9:-$EPS_PARAM}
ELEMENT_ORDER=${10:-1}

LCRACK=$(awk "BEGIN {print $WSTEG + $DHOLE}")
# Print LCRACK for debugging
echo "Calculated LCRACK: $LCRACK"

# Run the first Python script with the specified arguments
echo "Meshing effective stress RVE"
python3 mesh_effective_stiffness.py --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0"

if [ $? -ne 0 ]; then
  echo "Error: Meshing effective stiffness problem failed."
  exit 1
fi

echo "Running effective stiffness computation..."
python3 run_effective_stiffness.py --lam_micro_param "$LAM_MICRO_PARAM" --mue_micro_param "$MUE_MICRO_PARAM"

if [ $? -ne 0 ]; then
  echo "Error: Computing effective stiffness failed."
  exit 1
fi 

echo "Meshing fracture RVE..."
python3 mesh_fracture.py --nholes "$NHOLES" --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0"

if [ $? -ne 0 ]; then
  echo "Error: Meshing fracture problem failed."
  exit 1
fi

echo "Checking mesh data..."
python3 get_mesh_info.py --mesh_file "$MESH_FILE"

if [ $? -ne 0 ]; then
  echo "Error: Checking mesh data failed."
  exit 1
fi

echo "Running fracture simulation mpirun..."
mpirun -np 10 python3 run_simulation.py --mesh_file "$MESH_FILE" --in_crack_length "$LCRACK" --lam_micro_param "$LAM_MICRO_PARAM" --mue_micro_param "$MUE_MICRO_PARAM" --gc_micro_param "$GC_MICRO_PARAM" --eps_param "$EPS_PARAM" --element_order "$ELEMENT_ORDER"

# Check if the third script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: Third Python script failed."
  exit 1
fi

echo "All scripts completed successfully."