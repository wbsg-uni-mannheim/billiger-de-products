#!/bin/bash

#SBATCH --job-name=category_extraction
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/generate_categories_%j.out
#SBATCH --error=slurm_runs/logs/generate_categories_%j.err

# Activate your virtual environment
source .venv/bin/activate

# Run your Python script
python -u notebooks/category_extraction.py
