#!/bin/bash
#SBATCH --job-name=cross_wordcooc
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_wordcooc_%j.out
#SBATCH --error=slurm_runs/logs/cross_wordcooc_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -u -m src.models.wordcooc.run_cross_language
