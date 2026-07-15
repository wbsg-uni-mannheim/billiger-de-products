#!/bin/bash
#SBATCH --job-name=rsupcon_de
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

for category in products20cc80rnd000un products50cc50rnd000un products80cc20rnd000un; do
  for size in small medium large; do
    python -u src/models/r-supCon/run_finetune_siamese.py \
      --model_pretrained_checkpoint "results/generated/r-supcon/de/pretrain/${category}-${size}" \
      --do_train \
      --do_eval \
      --do_predict \
      --frozen=False \
      --train_file "data/processed/training-sets/preprocessed_${category}_train_${size}.pkl.gz" \
      --train_size="$size" \
      --validation_file "data/processed/validation-sets/preprocessed_${category}_valid_${size}.pkl.gz" \
      --test_file "data/processed/gold-standards_adjusted/preprocessed_${category}_gs.pkl.gz" \
      --evaluation_strategy=epoch \
      --tokenizer=roberta-base \
      --grad_checkpoint=True \
      --output_dir "results/generated/r-supcon/de/finetune/${category}-${size}/" \
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
  done
done
