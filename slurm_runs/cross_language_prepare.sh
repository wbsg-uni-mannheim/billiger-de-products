#!/bin/bash
#SBATCH --job-name=cross_language_prepare
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_language_prepare_%j.out
#SBATCH --error=slurm_runs/logs/cross_language_prepare_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

source slurm_runs/env.sh

# Both languages: the cross-language variants read the English released gold
# standards directly, and Blocks 2/3 rescore English cells from the same pickles.
python -u src/processing/prepare_pairs.py --language both
python -u src/processing/prepare_ditto_hiergat.py --language both
python -u -m src.processing.prepare_cross_language --seed 42

# Abort before any training if a derived test file does not carry the released
# labels. The July Ditto/HierGAT runs were scored against an inflated label vector.
python -u src/processing/verify_released_serialization.py --language both
