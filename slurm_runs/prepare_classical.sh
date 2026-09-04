#!/bin/bash
#SBATCH --job-name=prep_classical
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/prep_classical_%j.out
#SBATCH --error=slurm_runs/logs/prep_classical_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm_runs/env.sh

# WordCooc and Magellan read their own derived inputs. Without these the
# launchers glob an empty directory and exit 0 having done nothing.
python -u src/processing/prepare_wordcooc.py
BILLIGER_ENV=/home/aasteine/miniconda3/envs/entitymatch PATH="/home/aasteine/miniconda3/envs/entitymatch/bin:$PATH" \
  python -u src/processing/prepare_magellan.py

for d in data/processed/wordcooc/learning-curve data/processed_en/wordcooc/learning-curve; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "[check] $d -> $n files"
  [ "$n" -gt 0 ] || { echo "[check] FATAL: $d is empty"; exit 1; }
done
