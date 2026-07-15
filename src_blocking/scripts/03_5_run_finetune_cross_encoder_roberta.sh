#!/bin/bash
#SBATCH --job-name=03_5_run_finetune_cross_encoder_roberta
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/03_5_run_finetune_cross_encoder_roberta_%j.out
#SBATCH --error=slurm_runs/logs/03_5_run_finetune_cross_encoder_roberta_%j.err

BATCH=32
LR=5e-5
EPOCHS=20

echo "CUDA devices available:"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

export DATA_DIR=$(pwd)/data/blocking_short_desc
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

python src_blocking/contrastive/run_finetune_cross_encoder.py \
    --model_pretrained_checkpoint roberta-base \
    --do_train \
    --do_eval \
    --dataset_name products80cc20rnd050un \
    --train_file data/blocking/products80cc20rnd_050un/products80cc20rnd_050un-train.json.gz \
    --validation_file data/blocking/products80cc20rnd_050un/products80cc20rnd_050un-train.json.gz \
    --test_file data/blocking/products80cc20rnd_050un/products80cc20rnd_050un-gs.json.gz \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --tokenizer roberta-base \
    --output_dir src_blocking/reports/cross_encoder/products80cc20rnd050un-$BATCH-$LR-$EPOCHS-roberta-base/ \
    --per_device_train_batch_size $BATCH \
    --learning_rate $LR \
    --weight_decay 0.01 \
    --num_train_epochs $EPOCHS \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --fp16 \
    --metric_for_best_model loss \
    --dataloader_num_workers 4 \
    --disable_tqdm True
