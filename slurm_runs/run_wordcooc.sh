#!/bin/bash
#SBATCH --job-name=wordcooc
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/wordcooc_%j.out
#SBATCH --error=slurm_runs/logs/wordcooc_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -u src/models/wordcooc/run_wordcooc.py --language de
python -u src/models/wordcooc/run_wordcooc.py --language en
