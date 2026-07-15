#!/bin/bash
#SBATCH --job-name=gpt_en
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=110:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/gpt_en_%j.out
#SBATCH --error=slurm_runs/logs/gpt_en_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for cc in 20cc80 50cc50 80cc20; do
  for unseen in 000 050 100; do
    python -u src/models/gpt/gpt_batch_english.py --cc "$cc" --un "$unseen" --gptmodel gpt-5.2
    python -u src/models/gpt/gpt_batch_english_new_prompt.py --cc "$cc" --un "$unseen" --gptmodel gpt-5.2
  done
done
