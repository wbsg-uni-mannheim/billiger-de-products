#!/bin/bash
#SBATCH --job-name=magellan
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/magellan_%j.out
#SBATCH --error=slurm_runs/logs/magellan_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -u src/models/magellan/run_magellan.py --language de
python -u src/models/magellan/run_magellan.py --language en
