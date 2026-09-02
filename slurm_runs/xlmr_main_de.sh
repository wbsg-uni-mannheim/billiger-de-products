#!/bin/bash
#SBATCH --job-name=xlmr_de
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/xlmr_de_%j.out
#SBATCH --error=slurm_runs/logs/xlmr_de_%j.err

# XLM-R baseline on the recommended configuration (80 % corner cases), German version.
# Same cross-encoder pipeline, protocol and hyperparameters as the RoBERTa baseline
# (run_confidence_test.sh at batch size 32); only the backbone changes. One model per
# development set size, selected on the 000un validation split, evaluated on the Seen,
# Half-Seen and Unseen test sets. Per-pair predictions are written next to each run.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p slurm_runs/logs
source slurm_runs/env.sh

BACKBONE="xlm-roberta-base"
CATEGORY="products80cc20rnd000un"
SIZES="${SIZES:-small medium large}"
BATCH_SIZE=32
# XLM-R warms up more slowly than RoBERTa: with the shipped patience of 10 the
# early-stopping counter (F1=0.0 is the initial "best") killed most seeds at
# epoch 11 before they emitted a single true positive. 25 gives the backbone
# room to start learning. NOTE: the RoBERTa grid this is compared against ran
# at the default patience of 10, so XLM-R gets the larger epoch budget here.
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
echo "[xlmr] python=$(command -v python) host=$(hostname) date=$(date) backbone=$BACKBONE"

for size in $SIZES; do
  out="results/generated/xlmr/de/${CATEGORY}-${size}/"
  python -u -m src.cross_language.provenance \
    --output-dir "$out" \
    --model xlmr \
    --backbone "$BACKBONE" \
    --validation-file "data/processed/validation-sets/preprocessed_${CATEGORY}_valid_${size}.pkl.gz" \
    --train-file "data/processed/training-sets/preprocessed_${CATEGORY}_train_${size}.pkl.gz" \
    --seeds "${RERUN_SEEDS:-0,1,2}" \
    --batch-size "$BATCH_SIZE"

  python -u src/models/transformer_bert_confidence/run_finetune_baseline.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file "data/processed/training-sets/preprocessed_${CATEGORY}_train_${size}.pkl.gz" \
    --train_size="$size" \
    --validation_file "data/processed/validation-sets/preprocessed_${CATEGORY}_valid_${size}.pkl.gz" \
    --test_file "data/processed/gold-standards_adjusted/preprocessed_${CATEGORY}_gs.pkl.gz" \
    --evaluation_strategy=epoch \
    --tokenizer="$BACKBONE" \
    --grad_checkpoint=True \
    --output_dir "$out" \
    --per_device_train_batch_size="$BATCH_SIZE" \
    --learning_rate=5e-5 \
    --weight_decay=0.01 \
    --num_train_epochs=50 \
    --lr_scheduler_type=linear \
    --warmup_ratio=0.05 \
    --max_grad_norm=1.0 \
    --fp16 \
    --metric_for_best_model=f1 \
    --dataloader_num_workers=4 \
    --disable_tqdm=True \
    --save_strategy=epoch \
    --load_best_model_at_end=True \
    --augment=all
done
