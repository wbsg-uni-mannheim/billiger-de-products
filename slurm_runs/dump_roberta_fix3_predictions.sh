#!/bin/bash
#SBATCH --job-name=roberta_fix3_preds
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --output=slurm_runs/logs/roberta_fix3_preds_%j.out
#SBATCH --error=slurm_runs/logs/roberta_fix3_preds_%j.err

# Write the per-pair predictions that the roberta_fix3 runs never saved.
# Inference only: every model is loaded from its finished checkpoint, nothing is trained.

set -euo pipefail
# sbatch copies the script into its spool dir, so BASH_SOURCE cannot locate the repo.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
echo "working directory: $PWD"

PY=$HOME/miniconda3/envs/ditto-modern/bin/python
ROOT=results/generated/roberta_fix3

n_runs=$(find "$ROOT" -mindepth 4 -maxdepth 4 -name model.safetensors | wc -l)
echo "found $n_runs finished runs"
[ "$n_runs" -gt 0 ] || { echo "ERROR: no runs found under $ROOT from $PWD" >&2; exit 1; }
done_count=0

for lang in de en; do
  if [ "$lang" = "de" ]; then GS=data/processed; else GS=data/processed_en; fi
  for cell in "$ROOT/$lang"/*; do
    [ -d "$cell" ] || continue
    category=$(basename "$cell" | sed 's/-\(small\|medium\|large\)$//')
    for seed in "$cell"/*; do
      [ -f "$seed/model.safetensors" ] || continue
      echo "=== $seed ($lang, $category) ==="
      $PY -u src/models/transformer_bert_confidence/dump_predictions.py \
        --run_dir "$seed" \
        --test_file "$GS/gold-standards_adjusted/preprocessed_${category}_gs.pkl.gz" \
        --tokenizer roberta-base \
        --batch_size 64
      done_count=$((done_count + 1))
    done
  done
done

echo "processed $done_count of $n_runs runs"
[ "$done_count" -eq "$n_runs" ] || { echo "ERROR: not all runs processed" >&2; exit 1; }
echo "ALL DONE"
