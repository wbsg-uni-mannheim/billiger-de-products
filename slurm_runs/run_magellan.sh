#!/bin/bash
#SBATCH --job-name=magellan
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/magellan_%j.out
#SBATCH --error=slurm_runs/logs/magellan_%j.err

set -euo pipefail
# sbatch runs a copy out of /var/spool/slurmd; SLURM_SUBMIT_DIR points into the repo.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

# Magellan is the only matcher needing py_entitymatching.
BILLIGER_ENV=/home/aasteine/miniconda3/envs/entitymatch
source slurm_runs/env.sh

python -u src/models/magellan/run_magellan.py --language de
python -u src/models/magellan/run_magellan.py --language en
