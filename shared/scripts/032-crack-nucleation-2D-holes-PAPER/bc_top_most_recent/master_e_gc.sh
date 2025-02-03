#!/bin/bash

# Define parameter ranges or specific values
DHOLE_VALUES=(1.0)          # Example: hole sizes
WIDTH_VALUES=(8.0)          # Example: domain widths
HEIGHT_VALUES=(8.0)         # Example: domain heights
E0_VALUES=(0.02)            # Example: E0 values
E1_VALUES=(0.7)             # Example: E1 values
MESH_FILES=("mesh_adaptive_sym.xdmf")
EPS_VALUES=(0.1)            # Example: Epsilon values
ELEMENT_ORDER_VALUES=(1)    # Example: Element orders

# Define tuples of (E, Gc) such that E * Gc = 6.0
E_GC_PAIRS=(
  "6.0 1.0"   # E=6.0, Gc=1.0
  "3.0 2.0"   # E=3.0, Gc=2.0
  "2.0 3.0"   # E=2.0, Gc=3.0
  "1.5 4.0"   # E=1.5, Gc=4.0
  "1.2 5.0"   # E=1.2, Gc=5.0
)

# Define tuples of (E, Gc) such that E * Gc = 8.0
# E_GC_PAIRS=(
#   "8.0 1.0"   # E=6.0, Gc=1.0
#   "4.0 2.0"   # E=3.0, Gc=2.0
#   "2.0 4.0"   # E=2.0, Gc=3.0
#   "1.0 8.0"   # E=1.5, Gc=4.0
# )

# Define tuples of (E, Gc) such that E * Gc = 4.0
# E_GC_PAIRS=(
#   # "6.0 1.0"   # E=6.0, Gc=1.0
#   # "2.0 3.0"   # E=2.0, Gc=3.0
#   "1.5 4.0"   # E=1.5, Gc=4.0
# )

# Path to the script.sh (run_simulations.sh)
SCRIPT="./script.sh"

# Ensure the height_study folder exists
HEIGHT_STUDY_DIR="./e_gc_study"
mkdir -p "$HEIGHT_STUDY_DIR"

# Iterate over all combinations of parameters
for DHOLE in "${DHOLE_VALUES[@]}"; do
  for WIDTH in "${WIDTH_VALUES[@]}"; do
    for HEIGHT in "${HEIGHT_VALUES[@]}"; do
      for E0 in "${E0_VALUES[@]}"; do
        for E1 in "${E1_VALUES[@]}"; do
          for MESH_FILE in "${MESH_FILES[@]}"; do
            for PAIR in "${E_GC_PAIRS[@]}"; do
              E=$(echo $PAIR | awk '{print $1}')  # Extract E from the pair
              GC=$(echo $PAIR | awk '{print $2}')  # Extract Gc from the pair
              
              # Calculate Lamé parameters for the given E
              LAM=$(python3 -c "nu=0.3; E=${E}; print(${E}*nu/((1+nu)*(1-2*nu)))")
              MUE=$(python3 -c "nu=0.3; E=${E}; print(${E}/(2*(1+nu)))")

              for EPS in "${EPS_VALUES[@]}"; do
                for ELEMENT_ORDER in "${ELEMENT_ORDER_VALUES[@]}"; do
                  echo "Running simulation with parameters:"
                  echo "  dhole=$DHOLE, width=$WIDTH, height=$HEIGHT, e0=$E0, e1=$E1"
                  echo "  mesh_file=$MESH_FILE, E=$E, Gc=$GC, lambda=$LAM, mu=$MUE"
                  echo "  epsilon=$EPS, element_order=$ELEMENT_ORDER"

                  # Run the simulation script with the current parameters
                  $SCRIPT --dhole $DHOLE \
                          --width $WIDTH \
                          --height $HEIGHT \
                          --e0 $E0 \
                          --e1 $E1 \
                          --mesh_file $MESH_FILE \
                          --lam_param $LAM \
                          --mue_param $MUE \
                          --gc_param $GC \
                          --eps_param $EPS \
                          --element_order $ELEMENT_ORDER

                  # Check if the script succeeded
                  if [[ $? -ne 0 ]]; then
                    echo "Error: Simulation failed with parameters:"
                    echo "  dhole=$DHOLE, width=$WIDTH, height=$HEIGHT, e0=$E0, e1=$E1"
                    echo "  mesh_file=$MESH_FILE, E=$E, Gc=$GC, lambda=$LAM, mu=$MUE"
                    echo "  epsilon=$EPS, element_order=$ELEMENT_ORDER"
                    exit 1
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
done

# Move all folders starting with simulation_ into the height_study folder
for SIM_DIR in ./simulation_*; do
  # Check if the item exists and is a directory
  if [[ -e "$SIM_DIR" && -d "$SIM_DIR" ]]; then
    echo "Moving $SIM_DIR to $HEIGHT_STUDY_DIR"
    mv "$SIM_DIR" "$HEIGHT_STUDY_DIR"
  else
    echo "No directories matching 'simulation_*' found in ./scripts."
  fi
done

echo "All simulations completed successfully!"
