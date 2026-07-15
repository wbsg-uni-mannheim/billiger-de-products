#!/bin/bash

#SBATCH --job-name=generate-sets
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=../logs/generate_sets_%j.out
#SBATCH --error=../logs/generate_sets_%j.err

# Activate your virtual environment
source ../.venv/bin/activate

# Run your Python script
python generate-sets-final.py
