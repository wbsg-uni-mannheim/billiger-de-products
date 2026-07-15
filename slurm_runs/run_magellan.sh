#!/bin/bash

#SBATCH --job-name=run_magellan
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/run_magellan_%j.out
#SBATCH --error=slurm_runs/logs/run_magellan_%j.err

# Activate your virtual environment
source .venv/bin/activate

# Run your Python script
python -u src/models/magellan/run_magellan_with_codeCarbon_english.py
