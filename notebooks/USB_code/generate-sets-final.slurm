#!/bin/bash
#SBATCH --job-name=generate-sets             # Job name (appears in squeue)
#SBATCH --cpus-per-task=20                   # Number of CPU cores to use
#SBATCH --mem=30G                           # Memory allocation
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --time=30:00:00
#SBATCH --chdir=/work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/notebooks

# Activate your Python virtual environment
source ../.venv/bin/activate

# Run your Python script
python generate-sets-final.py
