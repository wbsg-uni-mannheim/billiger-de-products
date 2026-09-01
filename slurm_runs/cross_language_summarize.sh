#!/bin/bash
#SBATCH --job-name=cross_summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_summary_%j.out
#SBATCH --error=slurm_runs/logs/cross_summary_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

python -u -m src.summarize_cross_language_results
