#!/bin/bash
#SBATCH --job-name=cross_magellan
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_magellan_%j.out
#SBATCH --error=slurm_runs/logs/cross_magellan_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -u -m src.processing.prepare_cross_language_magellan
python -u -m src.models.magellan.run_cross_language
