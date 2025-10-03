#!/bin/bash
#SBATCH --job-name=clip_diffusion
#SBATCH --output=logs/clip_diffusion_%j.out
#SBATCH --error=logs/clip_diffusion_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1


# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"

# Load required modules 
module load cuda12.6/toolkit

# Activate virtual environment 
source /var/scratch/ciarella/xAI/.venv/bin/activate

# Run the Python script
python test.py

# Print completion time
echo "Completion time: $(date)"
