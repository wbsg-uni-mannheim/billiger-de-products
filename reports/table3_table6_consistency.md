# Table 3 vs. Table 6: why the DE-DE numbers disagree

Scope: the German `80cc20rnd` / `large` / `Half-Seen` cell of Table 3 and the
DE-DE column of Table 6 are supposed to be the same quantity — train on the
German `80cc20rnd000un` `large` training split, select on a German validation
split, evaluate on `products80cc20rnd050un_gs` (4,437 pairs), mean over three
seeds. They are not the same experiment.

Everything below was produced on branch `table3-table6-consistency`. Every
command is listed in [Commands](#commands-run).

---

## 1. Summary

| Matcher  | Table 3 (main) | Table 6 (cross-language, old) | Cause |
|----------|---------------:|------------------------------:|-------|
| WordCooc | 43.16 | 43.61 | different validation split **and** the reported classifier flips (LinearSVC vs. LogisticRegression) |
| Magellan | 36.42 | 36.34 | different validation split (both report XGBoost) |
| RoBERTa  | 59.29 | 63.44 | different validation split **and** different training batch size (32 vs. 1024) |
| R-SupCon | 47.27 | 48.49 | different validation split |
| HierGAT  | 65.40 | 60.79 | different validation split; additionally the published 65.40 is **not reproducible** from the checked-in HierGAT outputs, which give 66.13 |
| Ditto    | 64.10 | 65.36 | different validation split; additionally the published 64.10 is the mean over run_ids **0, 3, 4**, not over seeds 0, 1, 2 (40.53), and the cross-language 65.36 was a two-run mean over a stale `summary.csv` |

**The single dominant cause is the validation split**, and it is not the one the
leads assumed. See §2.

Lead 2 (HierGAT backbone) is **disproven**: both experiments already fine-tune
`roberta-base`. See §3.

Lead 3 (aggregation base) is **confirmed**, with concrete mechanisms for both
the WordCooc and the Ditto anomaly. See §4.

---

## 2. Root cause: the two experiments select on different validation splits

### 2.1 What each experiment actually uses

| | Main benchmark grid (Table 3) | Cross-language run (Table 6, old) |
|---|---|---|
| Training split | `products80cc20rnd000un_train_large` (26,571 pairs) | same |
| **Validation split** | **`products80cc20rnd000un_valid_large`** | **`products80cc20rnd050un_valid_large`** |
| Validation rows | **4,452** | **4,436** |
| Validation seen-product share | **1.00** | **0.50** |
| Test set | `products80cc20rnd050un_gs` (4,437) | identical file content, per variant |
| Backbone (RoBERTa, R-SupCon, Ditto, HierGAT) | `roberta-base` | `roberta-base` |
| Checkpoint-selection rule | epoch with the best validation F1 on the match class | same rule, different split |

Where that comes from, per matcher:

* **RoBERTa** — `src/models/transformer_bert_confidence/run_confidence_test.sh:19`
  passes `--validation_file data/processed/validation-sets/preprocessed_${category}_valid_${size}.pkl.gz`
  with `category=products80cc20rnd000un`. The old
  `slurm_runs/cross_language_roberta.sh` passed
  `data/processed_cross_language/validation-sets/preprocessed_products80cc20rnd050un_valid_large.pkl.gz`.
* **R-SupCon** — same pattern, `src/models/r-supCon/run_finetune_siamese.sh:22`.
* **Ditto** — `src/models/ditto/all_runs_de.py` passes no `--validation_file`, so
  `train_ditto.py:69` falls back to `configs.json`'s `validset`, which is
  `data/processed/ditto/data/final_output/preprocessed_products80cc20rnd000un_valid_large.txt`.
  The old cross-language script overrode it with the 050un file.
* **HierGAT** — identical structure, `src/models/hiergat/train.py` +
  `src/models/hiergat/task.json`.
* **WordCooc** — `src/processing/prepare_wordcooc.py` derives the validation
  file as `train_path.name.replace("train", "valid")`, i.e. the 000un split;
  `run_wordcooc.py` turns it into a `PredefinedSplit` for `RandomizedSearchCV`.
  The old `prepare_cross_language.prepare_wordcooc()` used the 050un split, so
  the cross-language run both fit a different `CountVectorizer` vocabulary and
  scored its hyper-parameter search on different rows.
* **Magellan** — same, via `prepare_cross_language_magellan.py`.

This cleanly explains the observed pattern. The matchers that *select* something
(RoBERTa, HierGAT, Ditto, R-SupCon pick a checkpoint) move the most; WordCooc and
Magellan only re-tune hyper-parameters on a grid whose optimum is fairly flat, so
they move by a few tenths.

Direct evidence that different checkpoints were selected — the RoBERTa
`trainer_state.json` of the two runs:

| Run | seed 0 | seed 1 | seed 2 | best validation F1 |
|---|---|---|---|---|
| cross-language (050un valid) | `checkpoint-208` | `checkpoint-468` | `checkpoint-442` | 0.7224 / 0.7203 / 0.7108 |
| main grid (000un valid, bs 32) | `checkpoint-7479` | `checkpoint-29916` | `checkpoint-32409` | 0.7784 / 0.8197 / 0.8111 |

Different splits, different selection signal, different checkpoints, different
reported test F1 — from the same training data.

### 2.2 Lead 1 as stated is wrong: the deduplication is a no-op

`prepare_cross_language.prepare_development_pairs()` removes every validation
pair whose unordered offer-id pair also occurs in training. **It removes zero
rows.** The released `products80cc20rnd050un_valid_large` split is already
pair-disjoint from training, exactly as `REPRODUCTION.md` §2 claims ("pair-disjoint
within each corner-case family under the priority Test > Validation > Train").

Measured:

```
train                        rows=26571 pos=12747 seen_product_share=1.0000 offer_pair_overlap_with_train=26571
valid_000un                  rows= 4452 pos=  485 seen_product_share=1.0000 offer_pair_overlap_with_train=0
valid_050un (released)       rows= 4436 pos=  490 seen_product_share=0.5000 offer_pair_overlap_with_train=0
valid_050un (cross, "dedup") rows= 4436 pos=  490 seen_product_share=0.5000 offer_pair_overlap_with_train=0
gs_050un (test)              rows= 4437 pos=  361 seen_product_share=0.5000 offer_pair_overlap_with_train=0
```

The deduplicated pickle is row-for-row equal to the released split, and the
serialized forms are byte-identical:

```
a03a05941648b6ea9f2c53fe695a3eb1  data/processed/hiergat/.../preprocessed_products80cc20rnd050un_valid_large.txt
a03a05941648b6ea9f2c53fe695a3eb1  data/processed_cross_language/hiergat/.../preprocessed_products80cc20rnd050un_valid_large.txt
a03a05941648b6ea9f2c53fe695a3eb1  data/processed/ditto/.../preprocessed_products80cc20rnd050un_valid_large.txt
a03a05941648b6ea9f2c53fe695a3eb1  data/processed_cross_language/ditto/.../preprocessed_products80cc20rnd050un_valid_large.txt
```

**Consequence for the paper: the 4,436 figure in §5.1 is correct.** It is the
released `80cc20rnd050un` `large` validation split, not a pre-deduplication
count. What is wrong is the surrounding claim that this split is what the
experiments select on. It is 4,452 for every number in Table 3, and it *was*
4,436 for every number in Table 6.

All assertions in `prepare_cross_language.py` were kept and all of them pass;
none were weakened. Two more were added for the selection split (pair-disjoint
from training, seen-product share exactly 1.0, serialized row count equal to the
pickle) and they pass as well.

### 2.3 Which number is correct under the protocol the paper describes

Under the protocol as written, **neither table's Half-Seen number is right, and
Table 6's is the closer of the two.**

`REPRODUCTION.md` §2 states: *"For supervised matchers, train on the corresponding
German or English `80cc20rnd000un` training file and validate on the same
language's `80cc20rnd050un` validation file"*, and *"Both `050un` and `100un`
experiments use a 50%-seen validation split; `100un` refers to the fully unseen
test condition, not to model selection."* §5 repeats it for the cross-language
run: selection on the *`80cc20rnd050un` large validation split*.

By that text the old Table 6 DE-DE column (RoBERTa 63.4, R-SupCon 48.5, HierGAT
60.8, Ditto 65.4) is the protocol-conforming number and Table 3's Half-Seen
column is not. Table 3 trains one model per (corner-case ratio, size) cell,
selects it on the 000un validation split, and then reads the Seen, Half-Seen and
Unseen test columns off that single checkpoint. The documented design needs two
training runs per cell — one selected on 000un for the Seen column, one selected
on 050un for the Half-Seen and Unseen columns — and the pipeline never did that.

**I did not act on this**, per the instruction to prefer aligning the
cross-language run and not to start a main-experiment redo without asking.
See §6 for the decision that was taken and §7 for what a redo would cost.

---

## 3. Lead 2 (HierGAT backbone): disproven

* `git status` on this working copy is clean; there is **no** uncommitted change
  to `src/models/hiergat/train.py` or `train_en.py`. The `--lm` argparse default
  was still `'bert'` in both files (and in `train_ditto.py` /
  `train_ditto_english.py`).
* That default was never reached. The main driver
  `src/models/hiergat/all_runs_de.py:27` passes `--lm roberta` explicitly, and so
  does `src/models/ditto/all_runs_de.py:29`. `slurm_runs/cross_language_hiergat.sh`
  passed `--lm roberta` too.
* The result filenames confirm it on both sides:
  `final_large_80cc20rnd000un_lr=5e-06_id=0_batch=16_lm=roberta_adjusted.txt`
  in `src/models/hiergat/output/` (main) and in
  `results/generated/cross_language/hiergat/` (cross-language).
* `src/models/hiergat/model/model.py:23` and `src/models/ditto/ditto_light/ditto.py:18`
  both map `roberta` to `roberta-base`.

**Every reported Ditto and HierGAT number, in both tables, fine-tunes
`roberta-base`.** The backbone is not a source of the discrepancy.

The unreachable `'bert'` default was still a live footgun, so it is now
`'roberta'` in all four trainers, and the backbone is recorded per result row
(§5).

---

## 4. Lead 3 (aggregation base): confirmed, with mechanisms

### 4.1 Ditto `runs=2`

`reports/cross_language/summary.csv` and `metrics.csv` were generated on 22 July
(`cross_summary_293594`), when only two Ditto result files existed. Four more
landed on 21–22 August and were never re-summarized:

| file | mtime | DE-DE F1 |
|---|---|---|
| `...id=1_adjusted_testset.txt` | Jul 18 07:47 | 63.75 |
| `...id=2_adjusted_testset.txt` | Jul 18 16:03 | 66.97 |
| `...id=3_adjusted_testset.txt` | Aug 21 19:13 | 15.05 |
| `...id=4_adjusted_testset.txt` | Aug 22 02:17 | 61.97 |
| `...id=5_adjusted_testset.txt` | Aug 22 02:29 | 63.76 |
| `...id=7_adjusted_testset.txt` | Aug 22 03:11 | 64.43 |

There is **no `id=0`**. Seed 0 was run twice — inside `cross_ditto_285169` and
again as the dedicated `cross_ditto_s0_291219` — and it collapsed both times.
`slurm_runs/logs/cross_ditto_s0_291219.out` ends with

```
[PRED DEBUG] mean prob= 0.4811243728085398 min= 0.4811243712902069 max= 0.4811244308948517 pos@0.5= 0 / 4436
epoch 50: dev_f1=0.19894437677628907, f1=0.1504793664026678, best_f1=0.1504793664026678
```

— a constant-output model, 0 predicted positives, dev F1 frozen for all 50 epochs.
Seeds 3, 4, 5, 6 and 7 were then run one at a time (`rerun_ditto_cross_seed*`),
seed 6 was cancelled, and no seed set was ever fixed. The August reruns also used
a **different conda environment** (`/ceph/.../home/kelagin/miniconda/envs/ditto_env_gpu`)
than the July runs (`/home/aasteine/miniconda3/envs/ditto-modern`).

The same pattern produced Table 3's Ditto cell. `src/models/ditto/output/` holds
five files for the 80cc20/large cell:

| run_id | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Half-Seen F1 | 69.06 | 32.52 | 20.00 | 60.73 | 62.55 |

`all_runs_de.py` only ever runs seeds 0–2, whose mean is **40.53**. The published
64.10 is the mean over run_ids **0, 3, 4** (64.11 here; the tiny residual is the
same drift discussed in §4.3). Seeds 1 and 2 collapsed and were silently replaced.
Nothing in the repository or the paper documents this.

### 4.2 WordCooc `runs=6` (and `runs=5` for NaiveBayes)

`run_wordcooc.run_wordcooc()` opens its report file with mode `"w"` once, then
appends one row per (run, classifier) — 3 runs x 6 classifiers = 18 rows. The
DE-DE file had **35**. Two `cross_language_wordcooc.sh` jobs ran against the same
output path concurrently: both truncated, the second truncation landed after the
first job had written the header and one row, and the remaining 36 rows
interleaved. 36 - 1 lost row = 35, and the lost row was NaiveBayes, the first
classifier of run 1 — which is exactly why NaiveBayes shows 5 and everything else
6. The other four language variants have 18 rows each because the second job did
not get that far.

This also silently changed which classifier WordCooc reports. Table 3 reports
LinearSVC (43.16, best in the main run); the old Table 6 reports
LogisticRegression (43.61, best in the cross-language run). Two different
classifiers under one row label.

`src/models/report_lock.py` now takes an advisory whole-file lock for the
duration of `run_wordcooc()` / `run_magellan()`. A sequential rerun still
truncates and rewrites as before; a concurrent one fails loudly.

### 4.3 Two findings outside the assignment, reported as required

**(a) Table 3's HierGAT cell is not reproducible from the checked-in outputs.**
`src/models/hiergat/output/` gives 65.86 / 65.75 / 66.77 → **66.13 ± 0.56**, but
the published cell is **65.40**. This is not a one-cell rounding issue: every one
of the nine German HierGAT cells in `output/analysis/hiergat_experiment_summary_de.xlsx`
is close to, but never equal to, the mean of the corresponding `.txt` files, and
`50cc50rnd000un/large` has only two of its three seed files on disk. The `.txt`
outputs in the repository are a different snapshot from the one Table 3 was built
from. RoBERTa, R-SupCon, WordCooc and Magellan all reproduce their Table 3 cells
exactly, so this is specific to HierGAT (and, via §4.1, to Ditto).

**(b) RoBERTa's Table 3 cell moved and the cross-language run did not follow.**
Commit `068a3fc` reran the whole RoBERTa grid at `per_device_train_batch_size=32`;
`results/generated/roberta_bs32_full/cell_means.csv` gives
`de,80cc20,large ... half_f1_mean 59.2945` — the 59.29 quoted in the task. The
older `bs=1024` reference reports give 60.56, which is still the number on the
project website. The cross-language job was still running `bs=1024`. Aligning the
validation split alone would therefore not have been enough for RoBERTa.

---

## 5. What changed in the code

| File | Change |
|---|---|
| `src/cross_language/common.py` | Adds `SELECTION_VARIANT` / `SELECTION_SEEN_SHARE` and path helpers pointing at the main pipeline's own files; keeps `VALIDATION_VARIANT` (050un) with `VALIDATION_SEEN_SHARE` for the alternative protocol; adds `EXPERIMENT_NAME = "de_train_000un_valid_000un_test_050un"`, which spells out all three splits so a result file cannot be mistaken for one from the other protocol. |
| `src/processing/prepare_cross_language.py` | Adds `prepare_selection_pairs()`: asserts the selection split is pair-disjoint from training, that its seen-product share is exactly 1.0, and that the serialized Ditto/HierGAT copies have the same row count. `prepare_development_pairs()` and **all** of its assertions are untouched. WordCooc features are now fit on `train + selection` instead of `train + 050un valid`. |
| `src/processing/prepare_cross_language_magellan.py`, `src/models/{wordcooc,magellan}/run_cross_language.py` | Use the selection split and the shared `EXPERIMENT_NAME`. |
| `src/models/report_lock.py` (new) | Advisory lock around the classical matchers' result tables (§4.2). |
| `src/models/{wordcooc/run_wordcooc.py,magellan/run_magellan.py}` | Acquire/release that lock; write through a single `report_path`. |
| `src/models/hiergat/train.py`, `train_en.py`, `src/models/ditto/train_ditto.py`, `train_ditto_english.py` | `--lm` default `'bert'` → `'roberta'`. Behaviour is unchanged (every driver passes it explicitly) but the unreachable default can no longer become a real mismatch. |
| `src/cross_language/provenance.py` (new) | Writes/reads a `provenance.json` per experiment: model, backbone, validation file path, its SHA-256 and row count, train file, seed set, batch size. |
| `src/summarize_cross_language_results.py` | Joins that record onto every result row. `metrics.csv` and `summary.csv` gain `backbone`, `validation_file`, `validation_rows`, `validation_sha256`. A summary group that mixes protocols keeps every distinct value joined by `\|`, so an aggregate over two setups cannot look clean. The script now also prints a `WARNING` when a group mixes protocols, when a group has no provenance, and when run counts differ across language variants. |
| `slurm_runs/cross_language_protocol.sh` (new) | One definition of the protocol (splits, backbone, seeds, batch sizes), sourced by every job; echoes the interpreter, host and selection split into the log. |
| `slurm_runs/cross_language_*.sh` | Source that file; write provenance before running; use the selection split; RoBERTa at `bs=32`. Also fixes `cd "$(dirname "${BASH_SOURCE[0]}")/.."`, which under `sbatch` resolves into `/var/spool/slurmd`, not the repository — replaced by `${SLURM_SUBMIT_DIR:-...}`. |

Resource requests (`--cpus-per-task`, `--mem`, `--time`, `--gres`, `--partition`)
are unchanged from the existing scripts.

Nothing under `data/solute_de/` or `data/solute_en/` was touched.

---

## 6. The protocol that is now applied

**Both experiments select on `products80cc20rnd000un_valid_large`** — 4,452
pairs, SHA-256 `83f0d76b…` (pickle) / `5e264fff…` (Ditto+HierGAT serialization),
seen-product share 1.0 — read directly out of `data/processed/`, i.e. literally
the same files the main grid uses. No copy is made, so the two experiments cannot
drift apart again.

| | value |
|---|---|
| Training split | `data/processed/training-sets/preprocessed_products80cc20rnd000un_train_large.pkl.gz` |
| Validation split | `data/processed/validation-sets/preprocessed_products80cc20rnd000un_valid_large.pkl.gz` (4,452 rows) |
| Backbone | `roberta-base` for RoBERTa, R-SupCon, Ditto and HierGAT |
| Seeds | 0, 1, 2 — all three reported, no replacement of a collapsed seed |
| WordCooc / Magellan runs | `random_state` 1, 2, 3 for all five language variants |
| Selection rule | epoch with the best validation F1 on the match class (`--metric_for_best_model=f1` / `best_dev_f1`) |
| Batch size | RoBERTa 32, R-SupCon 64, Ditto 64, HierGAT 16 — matching the main grid |
| Test sets | the five aligned `products80cc20rnd050un_gs_*` variants, 4,437 pairs each |

This is the "align the cross-language run to the main experiment" direction, as
instructed. The one caveat is stated plainly in §2.3: it makes the two tables
agree, but it makes them agree on a protocol that contradicts `REPRODUCTION.md`
§2 and §5 (and the corresponding paper text). Either the documentation is
corrected to say that selection uses the `000un` validation split throughout, or
the main experiment is redone — see §7.

**Explicit recommendation.** Selecting the Half-Seen and Unseen columns on a
100 %-seen validation split is the methodologically weaker choice: model
selection sees a distribution the test condition does not have. The
half-seen split is the better one and is what the paper already claims. Adopting
it means rerunning the main grid — 27 German variants plus 27 English ones, and
a second training run per cell if the Seen column is to keep its own 000un-selected
checkpoint. **I have not started that and will not without your go-ahead.**

## 7. Status of the reruns

Six jobs were submitted on the aligned protocol into
`results/generated/cross_language/`. The previous, mixed-protocol results were
moved to `results/generated/cross_language_legacy_050un_valid/` rather than
deleted, so the old Table 6 stays auditable.

| Job | ID | Expected wall time (from the previous runs) |
|---|---|---|
| `cross_language_wordcooc.sh` | 339742 | ~1.5 h |
| `cross_language_magellan.sh` | 339747 | ~1 h |
| `cross_language_roberta.sh` | 339743 | ~5 h |
| `cross_language_rsupcon.sh` | 339744 | ~4 h |
| `cross_language_ditto.sh` | 339745 | ~24 h |
| `cross_language_hiergat.sh` | 339746 | ~3.5 days |

`slurm_runs/cross_language_summarize.sh` has to be submitted after HierGAT
finishes; until then `metrics.csv` / `summary.csv` cover only the jobs that have
completed, and the summarizer prints a `WARNING` for any matcher whose language
variants have unequal run counts.

GPT-5.2 was **not** rerun: it is zero-shot, selects nothing, and its three
repetitions per variant are already consistent. A `provenance.json` recording
`backbone=gpt-5.2 (zero-shot)` and no validation file was written for it so its
rows are annotated like everything else.

## 8. Commands run

```bash
git checkout -b table3-table6-consistency

# --- diagnosis -------------------------------------------------------------
wc -l data/processed/hiergat/data/final_output/preprocessed_products80cc20rnd000un_valid_large.txt \
      data/processed/hiergat/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt \
      data/processed_cross_language/hiergat/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt \
      data/processed/hiergat/data/final_output/preprocessed_products80cc20rnd050un_gs.txt

md5sum data/processed/hiergat/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt \
       data/processed_cross_language/hiergat/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt \
       data/processed/ditto/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt \
       data/processed_cross_language/ditto/data/final_output/preprocessed_products80cc20rnd050un_valid_large.txt

# every measured figure in this report, recomputed from the repository
/home/aasteine/miniconda3/envs/ditto-modern/bin/python reports/table3_table6_consistency_checks.py

# --- realignment -----------------------------------------------------------
mkdir -p results/generated/cross_language_legacy_050un_valid
mv results/generated/cross_language/{wordcooc,magellan,roberta,r-supcon,ditto,hiergat} \
   results/generated/cross_language/{metrics.csv,summary.csv} \
   results/generated/cross_language_legacy_050un_valid/

rm -f data/processed_cross_language/wordcooc/preprocessed_products80cc20rnd050un_valid_large_wordcooc.pkl.gz
rm -f data/processed_cross_language/magellan/preprocessed_products80cc20rnd050un_{train,valid,trainonly}_large_cross_magellan_*

/home/aasteine/miniconda3/envs/ditto-modern/bin/python -u -m src.processing.prepare_cross_language --seed 42
# Prepared 5 cross-language test sets with 4437 pairs each
# Selection split data/processed/validation-sets/preprocessed_products80cc20rnd000un_valid_large.pkl.gz (products80cc20rnd000un): 4452 pairs
# Prepared 4436 disjoint DE-DE products80cc20rnd050un validation pairs

/home/aasteine/miniconda3/envs/ditto-modern/bin/python -u -m src.cross_language.provenance \
  --output-dir results/generated/cross_language/gpt --model gpt \
  --backbone "gpt-5.2 (zero-shot)" --seeds "3 repetitions, no training seed" --batch-size "n/a"

# --- reruns ----------------------------------------------------------------
source /home/aasteine/miniconda3/etc/profile.d/conda.sh
conda activate ditto-modern
sbatch slurm_runs/cross_language_wordcooc.sh   # 339742
sbatch slurm_runs/cross_language_roberta.sh    # 339743
sbatch slurm_runs/cross_language_rsupcon.sh    # 339744
sbatch slurm_runs/cross_language_ditto.sh      # 339745
sbatch slurm_runs/cross_language_hiergat.sh    # 339746
conda activate entitymatch
sbatch slurm_runs/cross_language_magellan.sh   # 339747

# --- after all six finish ---------------------------------------------------
conda activate ditto-modern
sbatch slurm_runs/cross_language_summarize.sh
```

A first submission round (339736–339741) failed immediately: the checked-in
`cd "$(dirname "${BASH_SOURCE[0]}")/.."` resolves to `/var/spool/slurmd` under
`sbatch`. Fixed in all `cross_language_*.sh` before the round above.
