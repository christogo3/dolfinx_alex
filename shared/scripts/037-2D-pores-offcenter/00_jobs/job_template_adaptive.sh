#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A p0023647
#SBATCH -t {TIME}  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu={MEMORY_VALUE}
#SBATCH -n {PROCESSOR_NUMBER}
#SBATCH -e /work/scratch/as12vapa/037-2D-pores-offcenter/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/037-2D-pores-offcenter/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# Set the working directory name
working_folder_name="{FOLDER_NAME}"  # Change this to your desired folder name
# Create the working directory under $HPC_SCRATCH
working_directory="$HPC_SCRATCH/037-2D-pores-offcenter/$working_folder_name"


# Default values for parameters
NHOLES={NHOLES} # needs to be int
WSTEG={WSTEG}
DHOLE={DHOLE}
E0={E0}
E1={E1}
MESH_FILE="{MESH_FILE}"
LAM_MICRO_PARAM={LAM_MICRO_PARAM}
MUE_MICRO_PARAM={MUE_MICRO_PARAM}
GC_MICRO_PARAM={GC_MICRO_PARAM}
CRACK_Y={CRACK_Y}

# Calculate EPS_PARAM as 6 times E0 using awk if not provided by user
EPS_PARAM={EPS_PARAM}
ELEMENT_ORDER={ELEMENT_ORDER}

LCRACK=$(awk "BEGIN {print $WSTEG + $DHOLE}")



# Navigate to $HPC_SCRATCH
cd $HPC_SCRATCH

srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/mesh_effective_stiffness.py --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0"

srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/run_effective_stiffness.py --lam_micro_param "$LAM_MICRO_PARAM" --mue_micro_param "$MUE_MICRO_PARAM"

srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/mesh_fracture_adaptive.py --nholes "$NHOLES" --dhole "$DHOLE" --wsteg "$WSTEG" --e0 "$E0" --e1 "$E1" --crack_y "$CRACK_Y"

srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/get_mesh_info.py --mesh_file "$MESH_FILE"

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n {PROCESSOR_NUMBER} apptainer exec --bind $HOME/dolfinx_alex/shared:/home,$working_directory:/work $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/run_simulation.py --mesh_file "$MESH_FILE" --in_crack_length "$LCRACK" --lam_micro_param "$LAM_MICRO_PARAM" --mue_micro_param "$MUE_MICRO_PARAM" --gc_micro_param "$GC_MICRO_PARAM" --eps_param "$EPS_PARAM" --element_order "$ELEMENT_ORDER" --crack_y "$CRACK_Y"

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE






