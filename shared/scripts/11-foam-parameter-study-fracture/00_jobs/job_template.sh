#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A project02338
#SBATCH -t 10080  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=6000
#SBATCH -n 64
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/11-foam-parameter-study-fracture/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/11-foam-parameter-study-fracture/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# # cd $HPC_SCRATCH
cd /home/as12vapa/dolfinx_alex

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 64 apptainer exec --bind ./shared:/home alex-dolfinx.sif python3 /home/scripts/11-foam-parameter-study-fracture/{FOLDER_NAME}/script.py \
    --mesh_file {MESH_FILE} \
    --lam_param {LAM_PARAM} \
    --mue_param {MUE_PARAM} \
    --Gc_param {GC_PARAM} \
    --eps_factor_param {EPS_FACTOR_PARAM} \
    --element_order {ELEMENT_ORDER}

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE



