#!/bin/bash
#SBATCH -A master # Replace with your account name
#SBATCH -p normal # Replace with the desired partition name
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --nodelist=hpc2
#SBATCH --output=cache/classificator_training.log
#SBATCH --error=cache/classificator_training.log
#SBATCH --open-mode=truncate

# Load modules
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0:

# Load any necessary modules or activate a virtual environment
source ~/skyline/skylineenv/bin/activate

# Commands to run your GPU job
python ./classificator_training.py
