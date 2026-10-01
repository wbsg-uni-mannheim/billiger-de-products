#!/bin/bash
#SBATCH --job-name=cross_rsupcon
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/cross_rsupcon_%j.out
#SBATCH --error=slurm_runs/logs/cross_rsupcon_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
source slurm_runs/cross_language_protocol.sh

pretrain_dir="results/generated/cross_language/r-supcon/pretrain/products80cc20rnd000un-large"
if [[ ! -f "$pretrain_dir/model.safetensors" ]]; then
  python -u src/models/r-supCon/run_pretraining.py \
    --do_train=True \
    --train_file "data/processed_cross_language/r-supcon/pretrain/products80cc20rnd000un_train_large.pkl.gz" \
    --id_deduction_set "data/solute_de/training-sets/products80cc20rnd000un_train_large.json.gz" \
    --tokenizer=roberta-base \
    --grad_checkpoint=True \
    --output_dir "$pretrain_dir/" \
    --temperature=0.07 \
    --per_device_train_batch_size=1024 \
    --learning_rate=5e-5 \
    --weight_decay=0.01 \
    --num_train_epochs=200 \
    --lr_scheduler_type=linear \
    --warmup_ratio=0.05 \
    --max_grad_norm=1.0 \
    --fp16 \
    --dataloader_num_workers=4 \
    --disable_tqdm=True \
    --save_strategy=epoch \
    --logging_strategy=epoch \
    --augment=all
fi

python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/r-supcon" \
  --model r-supcon \
  --backbone "$BACKBONE" \
  --validation-file "$SELECTION_VALIDATION_PKL" \
  --train-file "$TRAIN_PKL" \
  --seeds "0,1,2" \
  --batch-size "$RSUPCON_TRAIN_BATCH_SIZE"

# run_finetune_siamese.py loops seeds 0, 1, 2 internally and appends the seed
# to --output_dir.
python -u src/models/r-supCon/run_finetune_siamese.py \
  --model_pretrained_checkpoint "$pretrain_dir" \
  --do_train \
  --do_eval \
  --do_predict \
  --frozen=False \
  --train_file "$TRAIN_PKL" \
  --train_size=large \
  --validation_file "$SELECTION_VALIDATION_PKL" \
  --test_file "$TEST_PKL" \
  --cross_language_test_dir "$CROSS_TEST_PAIR_DIR" \
  --evaluation_strategy=epoch \
  --tokenizer=roberta-base \
  --grad_checkpoint=True \
  --output_dir "results/generated/cross_language/r-supcon/80cc20-large/" \
  --per_device_train_batch_size="$RSUPCON_TRAIN_BATCH_SIZE" \
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
  --load_best_model_at_end \
  --augment=all
