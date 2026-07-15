#!/bin/bash
#SBATCH --job-name=03_process_training_data
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/03_process_training_data_%j.out
#SBATCH --error=slurm_runs/logs/03_process_training_data_%j.err

export DATA_DIR=$(pwd)/data/blocking_benchmark_final
export PYTHONPATH=$(pwd)
source ~/miniconda/etc/profile.d/conda.sh
conda activate ditto_env_gpu

python src_blocking/contrastive_pretraining/processing/convert_ds_to_deepmatcher_format_training_data.py --size small --dataset products80cc20rnd --testset 050un --datadir $DATA_DIR

python src_blocking/contrastive_pretraining/processing/preprocess-deepmatcher-datasets.py
python src_blocking/contrastive_pretraining/contrastive/prepare-data-deepmatcher.py