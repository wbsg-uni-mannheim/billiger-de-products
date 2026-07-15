#!/bin/bash

#SBATCH --job-name=r_supCon_finetune
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=70G
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/r_supCon_finetune_%j.out
#SBATCH --error=slurm_runs/logs/r_supCon_finetune_%j.err

source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
which python

# --- Ensure we’re in the project root ---
cd /work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/

SIZES=("small" "medium" "large")
DATASETS=("products80cc20rnd000un" "products50cc50rnd000un" "products20cc80rnd000un")

export CUDA_VISIBLE_DEVICES=0 
MODEL="roberta-base" #"bert-base-german-cased"
CHECKPOINT=True
BATCH=1024
LR=5e-05
TEMP=0.07
FROZEN=False
AUG="all"
PREAUG=${10}

for CATEGORY in "${DATASETS[@]}"; do
	for SIZE in "${SIZES[@]}"; do
		python -u src/models/r-supCon/run_finetune_siamese_efficiency_tracker.py \
			--model_pretrained_checkpoint src/models/r-supCon/reports/contrastive/$CATEGORY-$SIZE-$AUG$BATCH-$LR-$TEMP-${MODEL##*/} \
			--do_train \
			--frozen=$FROZEN \
			--train_file data/processed/training-sets/preprocessed_${CATEGORY}_train_$SIZE.pkl.gz \
			--train_size=$SIZE \
			--validation_file data/processed/training-sets/preprocessed_${CATEGORY}_train_$SIZE.pkl.gz \
			--test_file data/processed/gold-standards_adjusted/preprocessed_${CATEGORY}_gs.pkl.gz \
			--evaluation_strategy=epoch \
			--tokenizer=$MODEL \
			--grad_checkpoint=$CHECKPOINT \
			--output_dir src/models/r-supCon/reports_de/contrastive-ft-siamese/$CATEGORY-$SIZE-$AUG$BATCH-$PREAUG$LR-$TEMP-$FROZEN-${MODEL##*/}_adjusted/ \
			--per_device_train_batch_size=64 \
			--learning_rate=$LR \
			--weight_decay=0.01 \
			--num_train_epochs=50 \
			--lr_scheduler_type="linear" \
			--warmup_ratio=0.05 \
			--max_grad_norm=1.0 \
			--fp16 \
			--metric_for_best_model=loss \
			--dataloader_num_workers=4 \
			--disable_tqdm=True \
			--save_strategy="epoch" \
			--load_best_model_at_end \
			--augment=$AUG 
			#--do_param_opt \
	done
done