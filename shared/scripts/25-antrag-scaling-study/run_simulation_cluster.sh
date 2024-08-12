#!/bin/bash
#SBATCH -J bench
#SBATCH -A project02338
#SBATCH -t 600  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=8000
#SBATCH -n 1
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/25-antrag-scaling-study/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/25-antrag-scaling-study/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# Set the working directory name
working_folder_name="n32"  # Change this to your desired folder name

# Create the working directory under $HPC_SCRATCH
working_directory="$HPC_SCRATCH/25-antrag-scaling-study/$working_folder_name"

mkdir -p $working_directory

# Navigate to $HPC_SCRATCH
cd $working_directory

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 1 apptainer exec --bind $HOME/dolfinx_alex/shared:/home --bind "$working_directory:/work" $HOME/dolfinx_alex/alex-dolfinx.sif python3 /home/scripts/25-antrag-scaling-study/script_combined.py \
    --mesh_file coarse_pores \
    --lam_param 1.0000 \
    --mue_param 1.0000 \
    --Gc_param 1.0 \
    --eps_factor_param 50.0 \
    --element_order 1

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE