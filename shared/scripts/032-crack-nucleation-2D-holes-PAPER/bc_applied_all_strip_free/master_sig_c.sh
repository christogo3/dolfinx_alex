#!/bin/bash

# Define parameter ranges or specific values
DHOLE_VALUES=(1.0)
WIDTH_VALUES=(8.0)
HEIGHT_VALUES=(4.5)
E0_VALUES=(0.02)
E1_VALUES=(0.7)
MESH_FILES=("mesh_adaptive_sym.xdmf")
# GC_VALUES=(0.5 1.0 2.0)            # Gc values
# EPS_VALUES=(0.05 0.1 0.2) # Epsilon values
# LAM_MUE_PAIRS=("0.5 0.5" "1.0 1.0" "2.0 2.0") # Lambda and Mu pairs

GC_VALUES=(0.8 1.5)            # Gc values
EPS_VALUES=(0.15 0.08) # Epsilon values
LAM_MUE_PAIRS=("1.0 1.0") # Lambda and Mu pairs

ELEMENT_ORDER_VALUES=(1)

# Desired sigma_c values
SIG_C_TARGETS=(0.5134898976610932 1.0269797953221864 2.053959590644373)  # Target sig_c values

# Path to the simulation script
SCRIPT="./script.sh"

# Iterate over parameter combinations and calculate sigma_c
for DHOLE in "${DHOLE_VALUES[@]}"; do
  for WIDTH in "${WIDTH_VALUES[@]}"; do
    for HEIGHT in "${HEIGHT_VALUES[@]}"; do
      for E0 in "${E0_VALUES[@]}"; do
        for E1 in "${E1_VALUES[@]}"; do
          for MESH_FILE in "${MESH_FILES[@]}"; do
            for PAIR in "${LAM_MUE_PAIRS[@]}"; do
              LAM=$(echo $PAIR | awk '{print $1}')
              MUE=$(echo $PAIR | awk '{print $2}')
              for GC in "${GC_VALUES[@]}"; do
                for EPS in "${EPS_VALUES[@]}"; do
                  for ELEMENT_ORDER in "${ELEMENT_ORDER_VALUES[@]}"; do
                    # Calculate sigma_c
                    SIG_C=$(awk -v mu="$MUE" -v gc="$GC" -v eps="$EPS" 'BEGIN { print 9 / 16 * sqrt((2.0 * mu * gc) / (6.0 * eps)) }')

                    # Check if sigma_c matches any target values
                    for TARGET_SIG_C in "${SIG_C_TARGETS[@]}"; do
                      SIG_C_DIFF=$(awk -v sig_c="$SIG_C" -v target="$TARGET_SIG_C" 'BEGIN { print (sig_c - target) }')
                      SIG_C_DIFF_ABS=$(awk -v diff="$SIG_C_DIFF" 'BEGIN { print (diff < 0 ? -diff : diff) }')

                      # Use awk for floating-point comparison
                      if awk "BEGIN {exit !( $SIG_C_DIFF_ABS < 0.005 )}"; then
                        echo "Running simulation for target sig_c=$TARGET_SIG_C:"
                        echo "  dhole=$DHOLE, width=$WIDTH, height=$HEIGHT, e0=$E0, e1=$E1"
                        echo "  mesh_file=$MESH_FILE, lambda=$LAM, mu=$MUE, gc=$GC, epsilon=$EPS, element_order=$ELEMENT_ORDER"
                        echo "  Computed sig_c=$SIG_C (target sig_c=$TARGET_SIG_C)"

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
                          echo "  mesh_file=$MESH_FILE, lambda=$LAM, mu=$MUE, gc=$GC, epsilon=$EPS, element_order=$ELEMENT_ORDER"
                          exit 1
                        fi
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
done

echo "All simulations completed successfully!"



