#!/bin/bash
#SBATCH --job-name=cross_ditto
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/cross_ditto_%j.out
#SBATCH --error=slurm_runs/logs/cross_ditto_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for seed in 0 1 2; do
  python -u src/models/ditto/train_ditto.py \
    --task final_large_80cc20rnd000un \
    --logdir src/models/ditto/results/ \
    --run_id "$seed" \
    --batch_size 64 \
    --max_len 256 \
    --lr 5e-5 \
    --n_epochs 50 \
    --finetuning \
    --lm roberta \
    --da del \
    --validation_file "data/processed_cross_language/ditto/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt" \
    --cross_language_test_dir "data/processed_cross_language/ditto/data/final_output" \
    --output_dir "results/generated/cross_language/ditto"
done
