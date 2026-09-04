# XLM-R baseline — recommended configuration and cross-language experiment

`xlm-roberta-base` run through the same cross-encoder pipeline, protocol and
evaluation as the RoBERTa baseline. Produced on branch `table3-table6-consistency`.

Contents of this directory are the **final** numbers (lr=2e-5). The runs at the
originally shipped lr=5e-5 are kept as evidence under
`results/generated/xlmr_lr5e-5/` and `results/generated/xlmr_lr5e-5_cross_language/`.

---

## 1. What was run

| | Table 3 (main grid) | Table 6 (cross-language) |
|---|---|---|
| launcher | `slurm_runs/xlmr_main_{de,en}.sh` | `slurm_runs/cross_language_xlmr.sh` |
| trains | DE and EN, sizes small/medium/large | German `large` only |
| models trained | 2 × 3 × 3 seeds = 18 | 3 seeds |
| evaluated on | Seen (000un), Half-Seen (050un), Unseen (100un) | 5 aligned variants of the 050un gold standard |
| results | `de/`, `en/` | `results/generated/cross_language/xlmr/` |

The five Table 6 columns are **inference passes** over the same three trained
models, not five separate trainings.

## 2. Configuration

Identical to the RoBERTa batch-32 grid (commit `068a3fc`) except the backbone
and the two optimiser settings marked below.

```
backbone            xlm-roberta-base
batch size          32
epochs              50
weight decay        0.01
scheduler           linear
max grad norm       1.0
fp16                yes
selection metric    validation F1 on the match class
learning rate       2e-5     <-- differs from RoBERTa (5e-5)
warmup ratio        0.1      <-- differs from RoBERTa (0.05)
early stopping      patience 25 (commit e1f675c; RoBERTa grid ran at 10)
seeds               0, 1, 2  (no substitutions)
```

**Why the learning rate differs.** At RoBERTa's 5e-5, XLM-R fails to leave the
all-negative solution on the smaller splits: 6 of 18 seed-runs finished at
exactly F1 = 0.0, with training loss pinned at ~0.4955, which is the entropy of
the label prior (~20 % positives). RoBERTa is stable at the same setting — its
worst cell is 0.130. The failures scale with training-set size (`large` never
failed, `small` worst), which is the classic XLM-R fine-tuning instability;
5e-5 sits above its usual 1e-5–3e-5 range. At 2e-5 with warmup 0.1 **every cell
converges and no seed was substituted**. Per-backbone LR selection on the
validation split is the intended reading of these numbers.

## 3. Consistency check (the point of this branch)

`reports/table3_table6_consistency.md` requires that the German
`80cc20rnd` / `large` / `Half-Seen` cell of Table 3 and the DE-DE column of
Table 6 be the same quantity.

| quantity | value |
|---|---|
| Table 3 — DE / Large / Half-Seen | **63.700865** (sd 0.256, seeds 0,1,2) |
| Table 6 — DE-DE | **63.700900** |
| difference | **0.000035** |

These come from two independent sets of SLURM jobs reached by different code
paths — the main grid rescores `baseline_predictions_un050.csv` against the
released gold standard, the cross-language row reads
`predict_cross_de_de_results.json`. Agreement to 3.5e-5 F1 confirms both that
the runs are reproducible from the seed and that the two pipelines compute the
same quantity.

This identity only holds because both tables use the same learning rate. At the
mixed configuration (main grid 2e-5, cross-language 5e-5) the gap was 2.2 F1.

## 4. Paper rows

Table 3 — F1, mean over 3 seeds, sd in brackets:

```
Small  & Seen      & 57.57 (2.86) & 57.83 (3.70) \\
Small  & Half-Seen & 51.32 (3.35) & 51.62 (4.47) \\
Small  & Unseen    & 43.31 (2.15) & 45.14 (2.97) \\
Medium & Seen      & 68.74 (0.98) & 71.85 (1.99) \\
Medium & Half-Seen & 60.38 (2.41) & 62.74 (1.73) \\
Medium & Unseen    & 50.28 (1.78) & 52.79 (1.35) \\
Large  & Seen      & 74.25 (0.25) & 72.87 (0.95) \\
Large  & Half-Seen & 63.70 (0.26) & 61.94 (1.27) \\
Large  & Unseen    & 51.35 (1.17) & 49.09 (1.64) \\
```

Table 6 — DE-DE F1 and difference to DE-DE, from unrounded means:

```
XLM-R & 63.7 & $-6.0$ & $-6.9$ & $-0.3$ & $-4.2$ \\
```

## 5. Per-seed results (eval F1 / predict F1)

| cell | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| de/small | 0.771 / 0.552 | 0.795 / 0.616 | 0.738 / 0.559 |
| de/medium | 0.803 / 0.698 | 0.818 / 0.689 | 0.788 / 0.675 |
| de/large | 0.823 / 0.743 | 0.826 / 0.739 | 0.825 / 0.745 |
| en/small | 0.771 / 0.585 | 0.775 / 0.620 | 0.713 / 0.530 |
| en/medium | 0.809 / 0.721 | 0.813 / 0.742 | 0.800 / 0.693 |
| en/large | 0.820 / 0.718 | 0.825 / 0.727 | 0.824 / 0.741 |

Cross-language, per seed, all on 4,437 pairs with 361 positives:

| seed | de_de | de_en | en_de | en_en | random |
|---|---|---|---|---|---|
| 0 | 63.50 | 57.81 | 56.28 | 63.17 | 60.08 |
| 1 | 64.06 | 57.73 | 57.14 | 63.34 | 59.52 |
| 2 | 63.54 | 57.49 | 56.91 | 63.54 | 59.05 |
| **mean** | **63.70** | **57.68** | **56.78** | **63.35** | **59.55** |

Full per-seed values: `XLM-R_main_per_seed.csv`, `XLM-R_cross_per_seed.csv`.

## 6. Summarizer warnings

The only warning emitted concerns the **RoBERTa** comparison grid, not XLM-R:

```
% WARNING roberta_bs32_full: cells with fewer than 3 seeds or seed std above 5 F1:
roberta_bs32_full de small Seen      38.39 sd 16.67  n_seeds 4  seeds 0,1,2,3
roberta_bs32_full de small Half-Seen 32.42 sd 15.69  n_seeds 4  seeds 0,1,2,3
roberta_bs32_full de small Unseen    30.08 sd 12.91  n_seeds 4  seeds 0,1,2,3
```

That is a pre-existing property of `results/generated/roberta_bs32_full`
(4 seeds, high variance on DE/small). No XLM-R cell triggered it.

## 7. Two harness defects found along the way

Both are in `src/models/transformer_bert_confidence/run_finetune_baseline.py`
and were worked around, not patched. They only bite when **rerunning into a
directory that already contains results**, which is why the final numbers here
were all produced in freshly created output directories, one seed per SLURM job.

1. **Silent resume** (`run_finetune_baseline.py:379-381`). Each seed calls
   `get_last_checkpoint(output_dir)` and passes the result to `trainer.train()`.
   A rerun into a populated directory resumes instead of retraining.
2. **Leaked early-stopping counter** (`:358`). One `EarlyStoppingCallback` is
   constructed outside the seed loop and shared by every seed in the process.
   In transformers 4.41.2 `on_train_begin` does not reset
   `early_stopping_patience_counter`.

Individually harmless; together they truncate runs. A fresh Trainer has
`best_metric = None`, which resets the counter on the first eval — but a
*resumed* Trainer loads `best_metric = 0.0`, so the reset never happens and the
inherited counter fires at once. Observed effect: a rerun where seed 0 trained
50 epochs while seeds 1 and 2 "trained" 2 epochs in 31 s each.

**Avoiding both:** wipe the seed's output directory before rerunning, and submit
one seed per job.

## 8. Reproducing

```bash
# Table 3 — one job per seed (18 jobs)
for lang in de en; do for size in small medium large; do for s in 0 1 2; do
  LR=2e-5 WARMUP=0.1 SIZES=$size RERUN_SEEDS=$s sbatch slurm_runs/xlmr_main_${lang}.sh
done; done; done

# Table 6 — one job per seed (3 jobs)
for s in 0 1 2; do
  LR=2e-5 WARMUP=0.1 RERUN_SEEDS=$s sbatch slurm_runs/cross_language_xlmr.sh
done

# Summaries
python src/models/transformer_bert_confidence/summarize_xlmr.py \
  --compare_root results/generated/roberta_bs32_full \
  --compare_cross_root results/generated/cross_language/roberta
bash slurm_runs/cross_language_summarize.sh
```

`LR`, `WARMUP` and `OUT_ROOT` are env overrides added to the launchers; their
defaults reproduce the shipped 5e-5 behaviour unchanged. `--compare_cross_root`
is needed for RoBERTa's Table 6 row — the script otherwise defaults it to a
nonexistent path and prints an empty block.

## 9. Wall-clock (final lr=2e-5 runs)

| job | cell | elapsed |
|---|---|---|
| 340567-340569 | de/small seeds 0-2 | ~00:26 each |
| 340570-340572 | de/medium seeds 0-2 | ~00:54 each |
| 340573-340575 | de/large seeds 0-2 | ~03:30 each |
| 340576-340578 | en/small seeds 0-2 | ~00:27 each |
| 340579-340581 | en/medium seeds 0-2 | ~00:53 each |
| 340582-340584 | en/large seeds 0-2 | 03:27, 05:08, 03:27 |
| 340635, 340636, 340682 | cross-language seeds 0-2 | 02:50, 02:29, 02:29 |

Two jobs failed and were resubmitted (340288, 340637), both with
`FileExistsError` on a TensorBoard run directory: its name is derived from the
job's **start** time and hostname, so two jobs starting in the same second on
the same node collide. Staggering submissions does not prevent this; resubmit
the affected seed alone.
