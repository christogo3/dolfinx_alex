#!/bin/bash

# ==============================================================================
# Script to convert meshes, copy templates, run simulations and postprocessing.
# Sets up a virtual display using Xvfb ONLY if DISPLAY is not already :99.
# ==============================================================================

# Start Xvfb only if DISPLAY is not set to :99
if [ "$DISPLAY" != ":99" ]; then
  echo "Starting virtual display on DISPLAY=:99"
  Xvfb :99 -screen 0 1024x768x24 &
  XVFB_PID=$!
  export DISPLAY=:99
  echo "Virtual display started with PID $XVFB_PID"
else
  echo "DISPLAY is already set to $DISPLAY, no need to start Xvfb"
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set paths relative to the script directory
SOURCE_DIR="$SCRIPT_DIR/00_template"
TARGET_DIR="$SCRIPT_DIR/meshes"
#MESH_INPUT_DIR="/home/resources/special-issue-2025/JM-25-24/1cube/JM-25-24_segmented_3D"  # <<< Set this to your actual mesh input directory
MESH_INPUT_DIR="/home/resources/special_issue_2025/JM-25-33_segmented_3D"

# Convert mesh files to DolfinX-compatible format
echo "Converting mesh files in: $MESH_INPUT_DIR"
python3 make_mesh_dlfx_compatible.py "$MESH_INPUT_DIR" -f mesh.xdmf || { echo "Mesh conversion failed"; exit 1; }

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist: $SOURCE_DIR"
  exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Target directory does not exist: $TARGET_DIR"
  exit 1
fi

# Iterate over all subfolders of the target directory
for subfolder in "$TARGET_DIR"/*/; do
  if [ -d "$subfolder" ]; then
    echo "=========================================="
    echo "Processing: $subfolder"
    echo "=========================================="

    echo "Copying template files to $subfolder"
    cp -v "$SOURCE_DIR"/* "$subfolder"

    # Change to the subfolder directory
    cd "$subfolder" || { echo "Failed to enter $subfolder"; exit 1; }

    # Run each step with error checking
    echo "Running: mpirun -n 10 python3 linearelastic_pressure_test.py"
    mpirun -n 10 python3 linearelastic_pressure_test.py std x || { echo "linearelastic_pressure_test.py failed"; exit 1; }


    # Go back to the original script directory
    cd "$SCRIPT_DIR"
    echo "Finished processing $subfolder"
    echo ""
  fi
done

echo "All done!"

# Clean up Xvfb if it was started by this script
if [ ! -z "$XVFB_PID" ]; then
  echo "Stopping Xvfb (PID $XVFB_PID)"
  kill $XVFB_PID
fi








