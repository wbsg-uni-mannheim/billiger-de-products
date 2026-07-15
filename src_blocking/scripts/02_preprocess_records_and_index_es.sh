#!/bin/bash
#SBATCH --job-name=blocking
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/02_preprocess_records_and_index_es_%j.out
#SBATCH --error=slurm_runs/logs/02_preprocess_records_and_index_es_%j.err

source ~/miniconda/etc/profile.d/conda.sh
ENV_NAME="blocking_env"
if conda activate "$ENV_NAME"; then
  echo "Activated conda environment: $ENV_NAME"
else
  echo "ERROR: Failed to activate conda environment '$ENV_NAME'"
  echo "Available conda envs:" || true
  conda env list || true
  exit 1
fi
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

cd /work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis

export DATA_DIR=$(pwd)/data/blocking_benchmark_final
export PYTHONPATH=$(pwd)

datasets=("small" "medium" "large")
for DATASET in "${datasets[@]}"; do
    python -m src_blocking.strategy.indexing.preprocess_records_and_index_es --dataset=$DATASET
done
