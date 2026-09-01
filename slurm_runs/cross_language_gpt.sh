#!/bin/bash
#SBATCH --job-name=cross_gpt
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm_runs/logs/cross_gpt_%j.out
#SBATCH --error=slurm_runs/logs/cross_gpt_%j.err

set -euo pipefail
# sbatch runs a copy of this file out of /var/spool/slurmd, so BASH_SOURCE does
# not point into the repository; SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

# GPT-5.2 is zero-shot: it selects nothing, so it has no validation file and is
# unaffected by the selection protocol the supervised matchers share.
python -u -m src.cross_language.provenance \
  --output-dir "results/generated/cross_language/gpt" \
  --model gpt \
  --backbone "gpt-5.2 (zero-shot)" \
  --seeds "3 repetitions, no training seed" \
  --batch-size "n/a"

for variant in de_de de_en en_de en_en random; do
  for prompt in simple rule_guided; do
    python -u -m src.models.gpt.gpt_batch_cross_language \
      --variant "$variant" \
      --prompt "$prompt" \
      --model gpt-5.2
  done
done
