#!/bin/bash

#SBATCH --job-name=gpt_german
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/gpt_german_%j.out
#SBATCH --error=slurm_runs/logs/gpt_german_%j.err

# Activate your virtual environment
source .venv/bin/activate

# Run your Python script
python -u src/models/gpt/gpt_batch_german.py --cc="80cc20" --un="100" --gptmodel="gpt-5.2"
