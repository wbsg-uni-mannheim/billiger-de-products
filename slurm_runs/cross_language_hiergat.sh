#!/bin/bash
#SBATCH --job-name=cross_hiergat
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/cross_hiergat_%j.out
#SBATCH --error=slurm_runs/logs/cross_hiergat_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm_runs/cross_language_protocol.sh

python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/hiergat" \
  --model hiergat \
  --backbone "$BACKBONE" \
  --validation-file "$SELECTION_VALIDATION_HIERGAT" \
  --train-file "data/processed/hiergat/data/final_output/preprocessed_${TRAIN_VARIANT}_train_${TRAIN_SIZE}.txt" \
  --seeds "0,1,2" \
  --batch-size "$HIERGAT_TRAIN_BATCH_SIZE"

for seed in $SEEDS; do
  python -u src/models/hiergat/train.py \
    --task final_large_80cc20rnd000un \
    --run_id "$seed" \
    --batch_size "$HIERGAT_TRAIN_BATCH_SIZE" \
    --max_len 256 \
    --lr 5e-6 \
    --n_epochs 50 \
    --finetuning \
    --split \
    --output_dir "results/generated/cross_language/hiergat" \
    --lm roberta \
    --validation_file "$SELECTION_VALIDATION_HIERGAT" \
    --cross_language_test_dir "$CROSS_TEST_HIERGAT_DIR"
done
