#!/bin/bash

# Hardcoded folder path - change this to your input folder
FOLDER="/home/scripts/052-Special-Issue-IJF-Hannover/resources/310125_var_bcpos_rho_10_120_004"

# Run the mesh conversion script
#python3 mesh2dlfxmesh.py "$FOLDER" 
# Run the phasefield script in parallel with 4 processes
mpirun -np 4 python3 phasefield_mbb_whole_folder.py "$FOLDER" 
