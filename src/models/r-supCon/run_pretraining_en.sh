#!/bin/bash

#SBATCH --job-name=r_supCon
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/r_supCon_%j.out
#SBATCH --error=slurm_runs/logs/r_supCon_%j.err

source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
which python

# --- Ensure we’re in the project root ---
cd /work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/

export CUDA_VISIBLE_DEVICES=0 

SIZES=("small" "medium" "large")
DATASETS=("products80cc20rnd000un" "products50cc50rnd000un" "products20cc80rnd000un")

MODEL="roberta-base" #"huawei-noah/TinyBERT_General_4L_312D" #"bert-base-cased"  #"bert-base-german-cased"
CHECKPOINT=True
BATCH=1024
LR=5e-05
TEMP=0.07
AUG="all"

for CATEGORY in "${DATASETS[@]}"; do
	for SIZE in "${SIZES[@]}"; do
		python -u src/models/r-supCon/run_pretraining_en.py \
			--do_train=True \
			--train_file data/processed_en/pre-train/$CATEGORY/${CATEGORY}_train_$SIZE.pkl.gz\
			--id_deduction_set data/derived_en/training-sets/${CATEGORY}_train_$SIZE.json.gz \
			--tokenizer=$MODEL \
			--grad_checkpoint=$CHECKPOINT \
			--output_dir src/models/r-supCon/reports_en/contrastive/$CATEGORY-$SIZE-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
			--temperature=$TEMP \
			--per_device_train_batch_size=$BATCH \
			--learning_rate=$LR \
			--weight_decay=0.01 \
			--num_train_epochs=200 \
			--lr_scheduler_type="linear" \
			--warmup_ratio=0.05 \
			--max_grad_norm=1.0 \
			--fp16 \
			--dataloader_num_workers=4 \
			--disable_tqdm=True \
			--save_strategy="epoch" \
			--logging_strategy="epoch" \
			--augment=$AUG 
	done
done