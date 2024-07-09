#!/bin/bash
#SBATCH -J cv_seg
#SBATCH -A project02338
#SBATCH -t 1200  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=6000
#SBATCH -n 48
#SBATCH -e /work/scratch/as12vapa/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

cd $HPC_SCRATCH

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 48 apptainer exec --bind $HOME/dolfinx_alex/shared:/home $HOME/dolfinx_alex/alex-dolfinx.sif python3 /home/scripts/17-voxel-image-segmentation/chan-veese-segmentation.py

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE