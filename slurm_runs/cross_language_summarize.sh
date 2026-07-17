#!/bin/bash
#SBATCH --job-name=cross_summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_summary_%j.out
#SBATCH --error=slurm_runs/logs/cross_summary_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -u -m src.summarize_cross_language_results
