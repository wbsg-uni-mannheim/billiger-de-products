#!/bin/bash

#SBATCH --job-name=run_wordcooc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/run_wordcooc_%j.out
#SBATCH --error=slurm_runs/logs/run_wordcooc_%j.err

# Activate your virtual environment
source .venv/bin/activate

# --- Diagnostics ---
echo "CUDA devices available:"
nvidia-smi

# Run your Python script
python src/models/wordcooc/run_wordcooc_codecarbon_english.py
