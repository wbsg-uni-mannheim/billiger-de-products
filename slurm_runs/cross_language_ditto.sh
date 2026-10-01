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
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm_runs/cross_language_protocol.sh

python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/ditto" \
  --model ditto \
  --backbone "$BACKBONE" \
  --validation-file "$SELECTION_VALIDATION_DITTO" \
  --train-file "data/processed/ditto/data/final_output/preprocessed_${TRAIN_VARIANT}_train_${TRAIN_SIZE}.txt" \
  --seeds "0,1,2" \
  --batch-size "$DITTO_TRAIN_BATCH_SIZE"

# Seeds 0, 1 and 2, all of them reported. A collapsed seed is a result, not a
# reason to roll another one: the published Ditto cell of the main table is the
# mean over run_ids 0, 3 and 4 because seeds 1 and 2 collapsed, which is why
# that cell cannot be reproduced from the documented seed set.
for seed in $SEEDS; do
  python -u src/models/ditto/train_ditto.py \
    --task final_large_80cc20rnd000un \
    --logdir src/models/ditto/results/ \
    --run_id "$seed" \
    --batch_size "$DITTO_TRAIN_BATCH_SIZE" \
    --max_len 256 \
    --lr 5e-5 \
    --n_epochs 50 \
    --finetuning \
    --lm roberta \
    --da del \
    --validation_file "$SELECTION_VALIDATION_DITTO" \
    --cross_language_test_dir "$CROSS_TEST_DITTO_DIR" \
    --output_dir "results/generated/cross_language/ditto"
done
