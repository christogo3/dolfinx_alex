#!/bin/bash
#SBATCH -J cv_seg
#SBATCH -A project02338
#SBATCH -t 1200  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=6000
#SBATCH -n 96
#SBATCH -e /work/scratch/as12vapa/cv_seg/%x.err.%j
#SBATCH -o /work/scratch/as12vapa/cv_seg/%x.out.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# Set the working directory name
working_folder_name="cv_seg"  # Change this to your desired folder name

# Create the working directory under $HPC_SCRATCH
working_directory="$HPC_SCRATCH/$working_folder_name"
mkdir -p "$working_directory"

# Navigate to $HPC_SCRATCH
cd $HPC_SCRATCH

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 96 apptainer exec --bind $HOME/dolfinx_alex/shared:/home --bind "$working_directory:/work" $HOME/dolfinx_alex/alex-dolfinx.sif python3 /home/scripts/17-voxel-image-segmentation/chan-veese-segmentation.py

EXITCODE=$?

# Exit the job script with the status of the scientific program
exit $EXITCODE
