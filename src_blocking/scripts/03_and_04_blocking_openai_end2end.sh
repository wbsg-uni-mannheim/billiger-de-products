#!/bin/bash
#SBATCH --job-name=blocking_openai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_runs/logs/04_blocking_openai_%j.out
#SBATCH --error=slurm_runs/logs/04_blocking_openai_%j.err

set -e

mkdir -p slurm_runs/logs

source ~/miniconda/etc/profile.d/conda.sh
conda activate blocking_env2

export PYTHONPATH=$(pwd)

DATASET=large
DATA_DIR=$(pwd)/data/blocking_benchmark_final/$DATASET
EMB_DIR=$(pwd)/data/blocking_benchmark_final/embeddings/openai/$DATASET
FAISS_DIR=$(pwd)/data/blocking_benchmark_final/faiss/$DATASET

mkdir -p "$EMB_DIR"
mkdir -p "$FAISS_DIR"

python -u src_blocking/embeddings_openai/run_openai_batch_embedding_and_faiss_indexing_parallel.py \
    --table_a "$DATA_DIR/tableA.csv" \
    --table_b "$DATA_DIR/tableB.csv" \
    --emb_dir "$EMB_DIR" \
    --faiss_dir "$FAISS_DIR"