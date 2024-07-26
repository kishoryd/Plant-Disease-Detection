#!/bin/bash
#SBATCH --job-name=plant_disease_multigpu    # Job name
#SBATCH --output=output_%j.log               # Output log file
#SBATCH --error=error_%j.log                 # Error log file
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:2                         # Number of GPUs per node
#SBATCH --time=24:00:00                      # Time limit hrs:min:sec
#SBATCH --partition=gpu                      # Partition name
#SBATCH --reservation=k_job                  # Reservation name

# Load necessary modules
module load DL/DL-CondaPy/3.7

# Activate the Conda environment
source activate 

conda activate tensor-flow_GPU

# Run the Python script
python plant_disease_multigpu.py
