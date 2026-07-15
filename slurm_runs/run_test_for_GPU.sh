#!/bin/bash

#SBATCH --job-name=gpu_test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/gpu_test%j.out
#SBATCH --error=slurm_runs/logs/gpu_test%j.err

# Initialize Conda (this line is critical)
source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate hier_env
which python

#export PYTHONPATH=$PYTHONPATH:$(pwd)/src/models/ditto

# --- Diagnostics ---
echo "CUDA devices available:"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run your Python script
python src/models/ditto/test_gpu.py
