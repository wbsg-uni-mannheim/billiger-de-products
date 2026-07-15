#!/bin/bash

#SBATCH --job-name=000_adjust_pred
#SBATCH --cpus-per-task=20
#SBATCH --mem=70G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/000_adjust_pred_%j.out
#SBATCH --error=slurm_runs/logs/000_adjust_pred_%j.err

# Initialize Conda (this line is critical)
source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
which python

# Run your Python script
python -u src/models/transformer_bert_confidence/test.py
