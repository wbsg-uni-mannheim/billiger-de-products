#!/bin/bash
#SBATCH --job-name=03_5_run_preetraining_deepmatcher
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/03_5_run_preetraining_deepmatcher_%j.out
#SBATCH --error=slurm_runs/logs/03_5_run_preetraining_deepmatcher_%j.err

BATCH=32
LR=5e-5
TEMP=0.07
EPOCHS=20
SERIALIZATION=$5
AUG=$6

echo "CUDA devices available:"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

#changed parameters: fp16=False, gradcheckpoint=False
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=$(pwd)/data/blocking_benchmark_final
export SIZE=medium

python src_blocking/contrastive_pretraining/contrastive/run_pretraining_deepmatcher.py \
    --do_train \
    --dataset_name=products80cc20rnd000un \
    --clean=True \
    --train_file data/blocking_benchmark_final/products80cc20rnd_050un/contrastive/products80cc20rnd_050un-train.pkl.gz \
    --id_deduction_set data/blocking_benchmark_final/products80cc20rnd_050un/products80cc20rnd_050un-train.json.gz \
    --tokenizer="roberta-base" \
    --grad_checkpoint=False \
    --output_dir src_blocking/reports_benchmark_final/contrastive/products80cc20rnd000un-$SIZE-clean-$AUG$BATCH-$LR-$TEMP-$EPOCHS-roberta-base-$SERIALIZATION/ \
    --temperature=$TEMP \
    --per_device_train_batch_size=$BATCH \
    --learning_rate=$LR \
    --weight_decay=0.01 \
    --num_train_epochs=$EPOCHS \
    --lr_scheduler_type="linear" \
    --warmup_ratio=0.05 \
    --max_grad_norm=1.0 \
    --fp16=False \
    --dataloader_num_workers=4 \
    --disable_tqdm=True \
    --save_strategy="epoch" \
    --logging_strategy="epoch" \
    --augment=$AUG \
    --serialization=$SERIALIZATION
