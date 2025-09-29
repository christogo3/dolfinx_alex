#!/bin/bash

# Hardcoded folder path - change this to your input folder
FOLDER="/home/scripts/054-Special-Issue-IJF-Hannover/resources/250925_TTO_mbb_festlager_var_a_E_var_min_max/mbb_festlager_var_a_E_min"

# Run the mesh conversion script
python3 mesh2dlfxmesh.py "$FOLDER" 
# Run the phasefield script in parallel with 4 processes
mpirun -np 4 python3 phasefield_mbb_whole_folder.py "$FOLDER" "min"
