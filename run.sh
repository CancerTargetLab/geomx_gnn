#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 0-10:00:00
#SBATCH -o /proj/berzelius-2023-241/users/x_mahei/geomx_gnn/out/run.log
#SBATCH -e /proj/berzelius-2023-241/users/x_mahei/geomx_gnn/out/run.log

# The '-A' SBATCH switch above is only necessary if you are member of several
# projects on Berzelius, and can otherwise be left out.

# Apptainer images can only be used outside /home. In this example the
# image is located here

module load Anaconda/2023.09-0-hpc1-bdist
conda activate geomx

# execute python script
python -m main --help