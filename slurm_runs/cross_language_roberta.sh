#!/bin/bash
#SBATCH --job-name=cross_roberta
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/cross_roberta_%j.out
#SBATCH --error=slurm_runs/logs/cross_roberta_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm_runs/cross_language_protocol.sh

python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/roberta" \
  --model roberta \
  --backbone "$BACKBONE" \
  --validation-file "$SELECTION_VALIDATION_PKL" \
  --train-file "$TRAIN_PKL" \
  --seeds "0,1,2" \
  --batch-size "$ROBERTA_TRAIN_BATCH_SIZE"

# run_finetune_baseline.py loops seeds 0, 1, 2 internally and appends the seed
# to --output_dir.
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
  --tokenizer=roberta-base \
  --grad_checkpoint=True \
  --output_dir "results/generated/cross_language/roberta/80cc20-large/" \
  --per_device_train_batch_size="$ROBERTA_TRAIN_BATCH_SIZE" \
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
