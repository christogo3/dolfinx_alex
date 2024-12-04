#!/bin/bash

# Define parameter ranges or specific values
DHOLE_VALUES=(1.0)          # Example: hole sizes
WIDTH_VALUES=(6.0 7.0 10.0 12.0)              # Example: domain widths
HEIGHT_VALUES=(4.5 )             # Example: domain heights
E0_VALUES=(0.02)               # Example: E0 values
E1_VALUES=(0.7)                 # Example: E1 values
MESH_FILES=("mesh_adaptive_sym.xdmf")
GC_VALUES=(1.0)             # Example: Gc values
EPS_VALUES=(0.1)               # Example: Epsilon values
ELEMENT_ORDER_VALUES=(1)          # Example: Element orders

# LAM_MUE_PAIRS=(                     # Pairs of lambda and mu
#   "0.75 0.75"
#   "1.0 1.0"
#   "1.5 1.5"
#   "2.0 2.0"
# )

LAM_MUE_PAIRS=(                     # Pairs of lambda and mu
  "1.0 1.0"
)


# Path to the script.sh (run_simulations.sh)
SCRIPT="./script.sh"

# Iterate over all combinations of parameters
for DHOLE in "${DHOLE_VALUES[@]}"; do
  for WIDTH in "${WIDTH_VALUES[@]}"; do
    for HEIGHT in "${HEIGHT_VALUES[@]}"; do
      for E0 in "${E0_VALUES[@]}"; do
        for E1 in "${E1_VALUES[@]}"; do
          for MESH_FILE in "${MESH_FILES[@]}"; do
            for PAIR in "${LAM_MUE_PAIRS[@]}"; do
              LAM=$(echo $PAIR | awk '{print $1}')  # Extract lambda from the pair
              MUE=$(echo $PAIR | awk '{print $2}')  # Extract mu from the pair
              for GC in "${GC_VALUES[@]}"; do
                for EPS in "${EPS_VALUES[@]}"; do
                  for ELEMENT_ORDER in "${ELEMENT_ORDER_VALUES[@]}"; do
                    echo "Running simulation with parameters:"
                    echo "  dhole=$DHOLE, width=$WIDTH, height=$HEIGHT, e0=$E0, e1=$E1"
                    echo "  mesh_file=$MESH_FILE, lambda=$LAM, mu=$MUE, gc=$GC"
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
                      echo "  mesh_file=$MESH_FILE, lambda=$LAM, mu=$MUE, gc=$GC"
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
done

echo "All simulations completed successfully!"

