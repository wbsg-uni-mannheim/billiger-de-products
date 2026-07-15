#!/bin/bash
#SBATCH --job-name=blocking
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/04_load_data_into_faiss_%j.out
#SBATCH --error=slurm_runs/logs/04_load_data_into_faiss_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate blocking_env2

# Load secrets (fail hard if missing)
SECRET_FILE="/work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/notebooks/env.elasticsearch"


if [ ! -f "$SECRET_FILE" ]; then
  echo "ERROR: Secret config not found: $SECRET_FILE"
  exit 1
fi

source "$SECRET_FILE"
# --- Elasticsearch VM credentials (from admin) ---
echo "Using Elasticsearch:"
echo "ES_HOST=$ES_HOST"
echo "ES_USER=$ES_USER"
echo "ES_INDEX=$ES_INDEX"

# optional: kurzer Connectivity-Check (auf VM statt localhost)
if ! curl -sf -u "$ES_USER:$ELASTICSEARCH_PASSWORD" "$ES_HOST" > /dev/null; then
  echo "ERROR: Elasticsearch VM not reachable or auth failed at $ES_HOST"
  exit 1
fi


export ES_INSTANCE="$ES_HOST"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)

export DATASET=medium
export MODELNAME=src_blocking/reports_benchmark_final/contrastive/products80cc20rnd000un-medium-clean-32-5e-5-0.07-20-roberta-base-
export DATA_DIR=$(pwd)/data/blocking_benchmark_final

python src_blocking/strategy/indexing/index_faiss_entity.py \
    --dataset=$DATASET \
    --bi_encoder_name='supcon_bi_encoder' \
    --model_name=$MODELNAME \
    --base_model='roberta-base' \
    --with_projection=False \
    --dimensions=768 \
    --serialization=attribute_names \
