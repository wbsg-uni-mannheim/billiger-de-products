#!/bin/bash

#SBATCH --job-name=translate_datasets
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/translate_datasets_%j.out
#SBATCH --error=slurm_runs/logs/translate_datasets_%j.err

# Activate your virtual environment
source .venv/bin/activate

# Run your Python script
python -u src/translate_to_english/transalte_datasets_to_english_multiple_batches.py
