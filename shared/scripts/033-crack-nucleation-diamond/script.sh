#!/bin/bash

# Default parameters for meshed_adaptive_sym.py
DEFAULT_DHOLE=1.0
DEFAULT_WIDTH=6.0
DEFAULT_HEIGHT=6.0
DEFAULT_E0=0.02
DEFAULT_E1=0.7

# Default parameters for run_simulation_K_sym.py
DEFAULT_MESH_FILE="mesh_adaptive_sym.xdmf"
DEFAULT_LAM_PARAM=1.0
DEFAULT_MUE_PARAM=1.0
DEFAULT_GC_PARAM=1.0
DEFAULT_EPS_PARAM=0.1
DEFAULT_ELEMENT_ORDER=1

# Parse arguments with defaults
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dhole) DHOLE="$2"; shift ;;
        --width) WIDTH="$2"; shift ;;
        --height) HEIGHT="$2"; shift ;;
        --e0) E0="$2"; shift ;;
        --e1) E1="$2"; shift ;;
        --mesh_file) MESH_FILE="$2"; shift ;;
        --lam_param) LAM_PARAM="$2"; shift ;;
        --mue_param) MUE_PARAM="$2"; shift ;;
        --gc_param) GC_PARAM="$2"; shift ;;
        --eps_param) EPS_PARAM="$2"; shift ;;
        --element_order) ELEMENT_ORDER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use defaults if arguments are not set
DHOLE="${DHOLE:-$DEFAULT_DHOLE}"
WIDTH="${WIDTH:-$DEFAULT_WIDTH}"
HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
E0="${E0:-$DEFAULT_E0}"
E1="${E1:-$DEFAULT_E1}"

MESH_FILE="${MESH_FILE:-$DEFAULT_MESH_FILE}"
LAM_PARAM="${LAM_PARAM:-$DEFAULT_LAM_PARAM}"
MUE_PARAM="${MUE_PARAM:-$DEFAULT_MUE_PARAM}"
GC_PARAM="${GC_PARAM:-$DEFAULT_GC_PARAM}"
EPS_PARAM="${EPS_PARAM:-$DEFAULT_EPS_PARAM}"
ELEMENT_ORDER="${ELEMENT_ORDER:-$DEFAULT_ELEMENT_ORDER}"

# Run the first Python script: meshed_adaptive_sym.py
echo "Running mesh_adaptive_sym.py with parameters:"
echo "--dhole $DHOLE --width $WIDTH --height $HEIGHT --e0 $E0 --e1 $E1"
python3 mesh_adaptive_sym.py --dhole $DHOLE --width $WIDTH --height $HEIGHT --e0 $E0 --e1 $E1
#python3 mesh_adaptive.py --dhole $DHOLE --width $WIDTH --height $HEIGHT --e0 $E0 --e1 $E1

# Check if the first script succeeded
if [[ $? -ne 0 ]]; then
    echo "Error: mesh_adaptive_sym.py failed to run. Exiting."
    exit 1
fi

# Run the second Python script: run_simulation_K_sym.py
echo "Running run_simulation_K_sym.py with parameters:"
echo "--mesh_file $MESH_FILE --lam_param $LAM_PARAM --mue_param $MUE_PARAM --gc_param $GC_PARAM --eps_param $EPS_PARAM --element_order $ELEMENT_ORDER"
mpirun -np 6 python3 run_simulation_K_sym.py --mesh_file $MESH_FILE --lam_param $LAM_PARAM --mue_param $MUE_PARAM --gc_param $GC_PARAM --eps_param $EPS_PARAM --element_order $ELEMENT_ORDER
#mpirun -np 10 python3 run_simulation_K.py --mesh_file $MESH_FILE --lam_param $LAM_PARAM --mue_param $MUE_PARAM --gc_param $GC_PARAM --eps_param $EPS_PARAM --element_order $ELEMENT_ORDER

# Check if the second script succeeded
if [[ $? -ne 0 ]]; then
    echo "Error: run_simulation_K_sym.py failed to run. Exiting."
    exit 1
fi

echo "Both scripts executed successfully!"
