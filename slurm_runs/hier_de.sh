#!/bin/bash
#SBATCH --job-name=hiergat_de
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=110:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/hiergat_de_%j.out
#SBATCH --error=slurm_runs/logs/hiergat_de_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u src/models/hiergat/all_runs_de.py
