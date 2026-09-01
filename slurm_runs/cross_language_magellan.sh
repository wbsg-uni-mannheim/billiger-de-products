#!/bin/bash
#SBATCH --job-name=cross_magellan
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_magellan_%j.out
#SBATCH --error=slurm_runs/logs/cross_magellan_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
# Magellan is the only matcher needing py_entitymatching.
BILLIGER_ENV=/home/aasteine/miniconda3/envs/entitymatch
source slurm_runs/cross_language_protocol.sh

python -u -m src.processing.prepare_cross_language_magellan

python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/magellan" \
  --model magellan \
  --backbone "n/a (Magellan similarity features)" \
  --validation-file "$SELECTION_VALIDATION_PKL" \
  --train-file "$TRAIN_PKL" \
  --seeds "1,2,3" \
  --batch-size "n/a"

python -u -m src.models.magellan.run_cross_language
