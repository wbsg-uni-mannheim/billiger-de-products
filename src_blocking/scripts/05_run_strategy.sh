#!/bin/bash
#SBATCH --job-name=05_run_strategy
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=150G
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/05_run_strategy_%j.out
#SBATCH --error=slurm_runs/logs/05_run_strategy_%j.err

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


export DATA_DIR=$(pwd)/data/blocking_benchmark_final
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0


python -u src_blocking/strategy/run_strategy.py --path_to_config=$(pwd)'/src_blocking/config/experiments/blocking_experiments_products80cc20rnd050un_medium.yml' --worker=1
