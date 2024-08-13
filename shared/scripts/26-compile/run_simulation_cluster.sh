#!/bin/bash
#SBATCH -J compile
#SBATCH -A project02338
#SBATCH -t 600  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=8000
#SBATCH -n 1
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/26-compile/%x.err.%j
#SBATCH -o /home/as12vapa/dolfinx_alex/shared/scripts/26-compile/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i02

# Create the working directory under $HPC_SCRATCH
working_directory="$HOME/dolfinx_alex"

mkdir -p $working_directory

# Navigate to $HPC_SCRATCH
cd $working_directory

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 1 apptainer build alex-dolfinx.sif apptainer.def

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE