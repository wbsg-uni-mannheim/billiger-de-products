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
cd "$(dirname "${BASH_SOURCE[0]}")/.."

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

python -u src/models/r-supCon/run_finetune_siamese.py \
  --model_pretrained_checkpoint "$pretrain_dir" \
  --do_train \
  --do_eval \
  --do_predict \
  --frozen=False \
  --train_file "data/processed/training-sets/preprocessed_products80cc20rnd000un_train_large.pkl.gz" \
  --train_size=large \
  --validation_file "data/processed_cross_language/validation-sets/preprocessed_products80cc20rnd050un_valid_large.pkl.gz" \
  --test_file "data/processed/gold-standards_adjusted/preprocessed_products80cc20rnd050un_gs.pkl.gz" \
  --cross_language_test_dir "data/processed_cross_language/gold-standards_adjusted" \
  --evaluation_strategy=epoch \
  --tokenizer=roberta-base \
  --grad_checkpoint=True \
  --output_dir "results/generated/cross_language/r-supcon/80cc20-large/" \
  --per_device_train_batch_size=64 \
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
