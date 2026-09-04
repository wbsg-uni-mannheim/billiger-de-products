#!/bin/bash
#SBATCH --job-name=cross_xlmr
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/cross_xlmr_%j.out
#SBATCH --error=slurm_runs/logs/cross_xlmr_%j.err

# XLM-R baseline for the cross-language experiment: identical to
# cross_language_roberta.sh except for the backbone. Trains on the German
# 80cc20rnd000un large split, selects on its 000un validation split and predicts
# the five aligned 80cc20rnd050un test variants (de_de, de_en, en_de, en_en, random).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p slurm_runs/logs
source slurm_runs/cross_language_protocol.sh
XLMR_BACKBONE="xlm-roberta-base"
XLMR_TRAIN_BATCH_SIZE="$ROBERTA_TRAIN_BATCH_SIZE"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-0.05}"

python -u -m src.cross_language.provenance \
  --output-dir "${OUT_ROOT:-results/generated/cross_language/xlmr}" \
  --model xlmr \
  --backbone "$XLMR_BACKBONE" \
  --validation-file "$SELECTION_VALIDATION_PKL" \
  --train-file "$TRAIN_PKL" \
  --seeds "${RERUN_SEEDS:-0,1,2}" \
  --batch-size "$XLMR_TRAIN_BATCH_SIZE"

python -u src/models/transformer_bert_confidence/run_finetune_baseline.py \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file "$TRAIN_PKL" \
  --train_size=large \
  --validation_file "$SELECTION_VALIDATION_PKL" \
  --test_file "$TEST_PKL" \
  --cross_language_test_dir "$CROSS_TEST_PAIR_DIR" \
  --evaluation_strategy=epoch \
  --tokenizer="$XLMR_BACKBONE" \
  --grad_checkpoint=True \
  --output_dir "${OUT_ROOT:-results/generated/cross_language/xlmr}/80cc20-large/" \
  --per_device_train_batch_size="$XLMR_TRAIN_BATCH_SIZE" \
  --learning_rate="$LR" \
  --weight_decay=0.01 \
  --num_train_epochs=50 \
  --lr_scheduler_type=linear \
  --warmup_ratio="$WARMUP" \
  --max_grad_norm=1.0 \
  --fp16 \
  --metric_for_best_model=f1 \
  --dataloader_num_workers=4 \
  --disable_tqdm=True \
  --save_strategy=epoch \
  --load_best_model_at_end=True \
  --augment=all
