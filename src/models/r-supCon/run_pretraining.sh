#!/bin/bash
#SBATCH --job-name=rsupcon_pretrain_de
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

for category in products20cc80rnd000un products50cc50rnd000un products80cc20rnd000un; do
  for size in small medium large; do
    python -u src/models/r-supCon/run_pretraining.py \
      --do_train=True \
      --train_file "data/processed/pre-train/$category/${category}_train_${size}.pkl.gz" \
      --id_deduction_set "data/solute_de/training-sets/${category}_train_${size}.json.gz" \
      --tokenizer=roberta-base \
      --grad_checkpoint=True \
      --output_dir "results/generated/r-supcon/de/pretrain/${category}-${size}/" \
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
  done
done
