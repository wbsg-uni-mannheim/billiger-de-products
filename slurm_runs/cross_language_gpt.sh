#!/bin/bash
#SBATCH --job-name=cross_gpt
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_gpt_%j.out
#SBATCH --error=slurm_runs/logs/cross_gpt_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for variant in de_de de_en en_de en_en random; do
  for prompt in simple rule_guided; do
    python -u -m src.models.gpt.gpt_batch_cross_language \
      --variant "$variant" \
      --prompt "$prompt" \
      --model gpt-5.2
  done
done
