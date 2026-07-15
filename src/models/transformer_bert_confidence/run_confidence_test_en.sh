#!/bin/bash

#SBATCH --job-name=run_roberta_confidece_finetune
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/run_roberta_confidece_finetune_%j.out
#SBATCH --error=slurm_runs/logs/run_roberta_confidece_finetune_%j.err

source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
which python


# --- Ensure we’re in the project root --- "products80cc20rnd000un" 
cd /work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/

SIZES=("small" "medium" "large")
DATASETS=("products80cc20rnd000un" "products50cc50rnd000un" "products20cc80rnd000un")
export CUDA_VISIBLE_DEVICES=0 
MODEL="roberta-base" #"bert-base-cased" #"bert-base-german-cased"
CHECKPOINT=True
BATCH=1024
LR=5e-05
AUG="all"

for CATEGORY in "${DATASETS[@]}"; do
	for SIZE in "${SIZES[@]}"; do
		python -u src/models/transformer_bert_confidence/run_finetune_baseline_en.py \
		--do_train \
		--train_file data/processed_en/training-sets/preprocessed_${CATEGORY}_train_$SIZE.pkl.gz \
		--train_size=$SIZE \
		--validation_file data/processed_en/training-sets/preprocessed_${CATEGORY}_train_$SIZE.pkl.gz \
		--test_file data/processed_en/gold-standards_adjusted/preprocessed_${CATEGORY}_gs.pkl.gz \
		--evaluation_strategy=epoch \
		--tokenizer=$MODEL \
		--grad_checkpoint=$CHECKPOINT \
		--output_dir src/models/transformer_bert_confidence/reports_en/baseline/$CATEGORY-$SIZE-$AUG$BATCH-$LR-${MODEL##*/}/ \
		--per_device_train_batch_size=$BATCH \
		--learning_rate=$LR \
		--weight_decay=0.01 \
		--num_train_epochs=50 \
		--lr_scheduler_type="linear" \
		--warmup_ratio=0.05 \
		--max_grad_norm=1.0 \
		--fp16 \
		--metric_for_best_model=f1 \
		--dataloader_num_workers=4 \
		--disable_tqdm=True \
		--save_strategy="epoch" \
		--load_best_model_at_end=True \
		--augment=$AUG
		#--do_param_opt \
	done
done