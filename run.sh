#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 0-10:00:00
#SBATCH -o /proj/berzelius-2023-241/users/$(id -un)/geomx_gnn/output_file.log
#SBATCH -e /proj/berzelius-2023-241/users/$(id -un)/geomx_gnn/error_file.log

# The '-A' SBATCH switch above is only necessary if you are member of several
# projects on Berzelius, and can otherwise be left out.

# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-2023-241/users/$(id -un)/geomx_gnn

module load Anaconda/2021.05-nsc1
conda activate geomx

# execute python script
python -m main