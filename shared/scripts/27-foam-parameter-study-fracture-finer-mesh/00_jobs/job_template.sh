#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A project02338
#SBATCH -t 10080  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=6000
#SBATCH -n 64
#SBATCH -e /work/scratch/as12vapa/19-foam-parameter-study-fracture-finer-mesh/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/19-foam-parameter-study-fracture-finer-mesh/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# Set the working directory name
working_folder_name="{FOLDER_NAME}"  # Change this to your desired folder name

# Create the working directory under $HPC_SCRATCH
working_directory="$HPC_SCRATCH/19-foam-parameter-study-fracture-finer-mesh/$working_folder_name"
mkdir -p "$working_directory"

# Navigate to $HPC_SCRATCH
cd $HPC_SCRATCH

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 64 apptainer exec --bind $HOME/dolfinx_alex/shared:/home --bind "$working_directory:/work" $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/script.py \
    --mesh_file {MESH_FILE} \
    --lam_param {LAM_PARAM} \
    --mue_param {MUE_PARAM} \
    --Gc_param {GC_PARAM} \
    --eps_factor_param {EPS_FACTOR_PARAM} \
    --element_order {ELEMENT_ORDER}

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE



