#SBATCH --job-name=clip_diffusion_v2
#SBATCH --output=logs/clip_diffusion_v2_%j.out
#SBATCH --error=logs/clip_diffusion_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1

# Check if config argument is provided
if [ -z "$1" ]; then
  echo "Usage: sbatch submit_train.sh <config.yaml>"
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment 
source ~/.virtualenvs/xAI/bin/activate

# Run the Python script
python train.py --config $1

# Print completion time
echo "Completion time: $(date)"
