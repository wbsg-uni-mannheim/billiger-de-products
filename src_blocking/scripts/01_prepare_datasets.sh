#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/01_prepare_datasets_%j.out
#SBATCH --error=slurm_runs/logs/01_prepare_datasets_%j.err

source ~/miniconda/etc/profile.d/conda.sh
# Activate your environment
conda activate ditto_env_gpu
cd /work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis

export DATA_DIR=$(pwd)/data/blocking_benchmark_final

export PYTHONPATH=$(pwd)

echo $PYTHONPATH

# List of datasets
datasets=("small" "medium" "large")

for DATASET in "${datasets[@]}"
do
    python -m src_blocking.data.convert_table_to_query_table --dataset=$DATASET --table_name=tableA
    python -m src_blocking.data.convert_table_to_table_format --dataset=$DATASET --table_name=tableB


done