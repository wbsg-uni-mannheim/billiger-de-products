#!/bin/bash

#SBATCH --job-name=run_ditto
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/run_ditto%j.out
#SBATCH --error=slurm_runs/logs/run_ditto%j.err

# Initialize Conda (this line is critical)
source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
which python

#export PYTHONPATH=$PYTHONPATH:$(pwd)/src/models/ditto

# --- Diagnostics ---
echo "CUDA devices available:"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your Python script
python -u src/models/ditto/all_runs.py
