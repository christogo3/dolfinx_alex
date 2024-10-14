#!/bin/bash

# Default values for parameters
L0=${1:-4.0}
WSTEG=${2:-0.4}
HSTEG=${3:-0.5}
WHOLE=${4:-0.2}
MESH_FILE=${5:-"mesh_holes.xdmf"}
LAM_PARAM=${6:-1.0}
MUE_PARAM=${7:-1.0}
GC_PARAM=${8:-1.0}
EPS_FACTOR_PARAM=${9:-0.05}
ELEMENT_ORDER=${10:-1}

# Run the first Python script with the specified arguments
echo "Running first Python script..."
python3 mesh_holes.py --l0 "$L0" --wsteg "$WSTEG" --hsteg "$HSTEG" --whole "$WHOLE"

# Check if the first script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: First Python script failed."
  exit 1
fi

echo "Running second Python script..."
python3 get_mesh_info.py --mesh_file "$MESH_FILE"

# Check if the second script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: Second Python script failed."
  exit 1
fi

# Run the third Python script using mpirun with the specified arguments
echo "Running third Python script with mpirun..."
mpirun -np 8 python3 run_simulation.py --mesh_file "$MESH_FILE" --lam_param "$LAM_PARAM" --mue_param "$MUE_PARAM" --Gc_param "$GC_PARAM" --eps_factor_param "$EPS_FACTOR_PARAM" --element_order "$ELEMENT_ORDER"

# Check if the third script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: Third Python script failed."
  exit 1
fi

echo "All scripts completed successfully."

