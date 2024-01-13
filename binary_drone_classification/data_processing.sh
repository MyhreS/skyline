#!/bin/bash
#SBATCH -A master # Replace with your account name
#SBATCH -p normal # Replace with the desired partition name
#SBATCH --nodelist=hpc2 # Use node hpc2
#SBATCH --output=cache/data_processing.log
#SBATCH --error=cache/data_processing.log
#SBATCH --open-mode=truncate


# Load any necessary modules or activate a virtual environment
source ~/skyline/skylineenv/bin/activate

# Commands to run your GPU job
python ./data_processing.py
