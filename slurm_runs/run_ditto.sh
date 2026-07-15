#!/bin/bash
#SBATCH --job-name=ditto_en
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/ditto_en_%j.out
#SBATCH --error=slurm_runs/logs/ditto_en_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u src/models/ditto/all_runs.py
