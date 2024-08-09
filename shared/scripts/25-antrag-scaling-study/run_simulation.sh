#!/bin/bash

# Define parameters
MESH_FILE="coarse_pores"
LAM_PARAM=1.0
MUE_PARAM=1.0
GC_PARAM=2.7974
EPS_FACTOR_PARAM=17.8739
ELEMENT_ORDER=1

# Run the Python script with mpirun
mpirun -np 10 python3 script_combined.py --mesh_file $MESH_FILE \
                                          --lam_param $LAM_PARAM \
                                          --mue_param $MUE_PARAM \
                                          --Gc_param $GC_PARAM \
                                          --eps_factor_param $EPS_FACTOR_PARAM \
                                          --element_order $ELEMENT_ORDER
