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
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for seed in 0 1 2; do
  python -u src/models/hiergat/train.py \
    --task final_large_80cc20rnd000un \
    --run_id "$seed" \
    --batch_size 16 \
    --max_len 256 \
    --lr 5e-6 \
    --n_epochs 50 \
    --finetuning \
    --split \
    --output_dir "results/generated/cross_language/hiergat" \
    --lm roberta \
    --validation_file "data/processed_cross_language/hiergat/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt" \
    --cross_language_test_dir "data/processed_cross_language/hiergat/data/final_output"
done
