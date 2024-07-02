#!/bin/bash
#SBATCH -J offset_crack
#SBATCH -A project02338
#SBATCH -t 10080  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=6000
#SBATCH -n 48
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/13-fracture-foam-offset-crack/medium/%x.err.%j
#SBATCH -e /home/as12vapa/dolfinx_alex/shared/scripts/13-fracture-foam-offset-crack/medium/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# # cd $HPC_SCRATCH
cd /home/as12vapa/dolfiny_alex

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 48 apptainer exec --bind ./shared:/home alex-dolfinx.sif python3 /home/scripts/13-fracture-foam-offset-crack/medium/pfmfrac.py

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE