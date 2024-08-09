#!/bin/bash
#SBATCH -J scal
#SBATCH -A project02338
#SBATCH -t 10080  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=9000
#SBATCH -n {PROCESSOR_NUMBER}
#SBATCH -e /work/scratch/as12vapa/25-antrag-scaling-study/{FOLDER_NAME}/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/25-antrag-scaling-study/{FOLDER_NAME}/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01


# Create the working directory under $HPC_SCRATCH
working_directory="$HPC_SCRATCH/25-antrag-scaling-study"

# Navigate to $HPC_SCRATCH
cd $HPC_SCRATCH

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n {PROCESSOR_NUMBER} apptainer exec --bind $HOME/dolfinx_alex/shared:/home --bind "$working_directory:/work" $HOME/dolfinx_alex/alex-dolfinx.sif python3 $working_directory/script_combined.py \
    --mesh_file medium_pores \
    --lam_param 1.0 \
    --mue_param 1.0 \
    --Gc_param 0.9684 \
    --eps_factor_param 51.6336 \
    --element_order 1

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE






