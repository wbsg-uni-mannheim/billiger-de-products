# Reproduction Guide — German Product Entity Matching Benchmark

This document is the single index for **where every piece of code, configuration, prompt,
and result lives** so the benchmark experiments can be reproduced and published.

Repo root (all paths are relative to it):
`Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis/`

> **⚠️ Publishing note.** Almost everything under `data/` and every `results*/`, `reports*/`,
> `output*/`, `model_output*/`, `checkpoints*/` folder is **git-ignored** (see `.gitignore`).
> So the *code* is tracked, but the *result files and datasets listed below exist only on disk*
> and are **not** currently committed. Decide per section what to publish (see
> [§10 Publishing checklist](#10-publishing-checklist)).
>
> **⚠️ Secret.** `src/models/gpt/secret_config.py` and
> `src_blocking/embeddings_openai/secret_config.py` contain a hard-coded `OPENAI_API_KEY`.
> Both files are git-ignored (verified), so they will not be published — but the key should be
> **rotated** and the scripts changed to read `os.environ["OPENAI_API_KEY"]` before release.

---

## 0. The benchmark at a glance

- **Task:** binary product entity matching (pair → match / non-match).
- **Languages:** German (original) and English (machine-translated). Language is encoded by the
  *directory*, not the filename: German = `data/derived/`, `data/processed/`, model dirs with no
  suffix; English = `*_en` directories.
- **27 variants** per language = **3 corner-case shares × 3 training sizes × 3 test-difficulty
  levels**:
  - corner-case share: `20cc80rnd`, `50cc50rnd`, `80cc20rnd` (cc% corner cases / rnd% random)
  - training size: `small`, `medium`, `large`
  - test difficulty (unseen ratio of the gold standard): `000un` (seen), `050un` (half-seen),
    `100un` (unseen)
- **Three random seeds** per neural model (0, 1, 2); classical models use global seed 42 + runs
  1/2/3. Reported numbers are the **mean over the three seeds**.
- **Models benchmarked:** WordCooc, Magellan, RoBERTa baseline, R-SupCon, HierGAT, Ditto, GPT-5.2.

**Key insight on file grouping:** train and validation sets are *always* `000un`; only the
gold-standard/test file changes across seen/half-seen/unseen. One trained model is evaluated
against the `000un`, `050un`, and `100un` gold standards in turn.

---

## 1. Repository structure map

```
src/
  processing/                     # data preparation & per-model formatting
  translate_to_english/           # DE→EN translation pipeline (OpenAI batch)
  models/
    wordcooc/                     # WordCooc baseline
    magellan/                     # Magellan baseline
    transformer_bert_confidence/  # RoBERTa baseline (+ calibration study)
    r-supCon/                     # R-SupCon (contrastive pretrain + Siamese finetune)
    hiergat/                      # HierGAT
    ditto/                        # Ditto
    gpt/                          # GPT-5.2 LLM matcher + analysis
  efficiency_tracking.py          # CodeCarbon helper
src_blocking/                     # blocking benchmark (candidate generation)
slurm_runs/                       # SLURM launch scripts for all models
environments/                     # conda envs + requirements.txt
data/                             # datasets, batch files, results (GIT-IGNORED)
data_analysis/                    # dataset statistics / category analysis
notebooks/                        # dataset generation notebooks + category work
old_items/                        # OLD / superseded artifacts — do not use
```

---

## 2. Installation

Python **3.10.14** for the deep-learning models; **3.9.21** for Magellan.

| Environment file | Used by | Python | Install |
|---|---|---|---|
| `environments/ditto_env_gpu.yml` | Ditto, R-SupCon, RoBERTa baseline | 3.10.14 | `conda env create -f environments/ditto_env_gpu.yml` → `conda activate ditto_env_gpu` |
| `environments/hier_env.yml` | HierGAT | 3.10.14 | `conda env create -f environments/hier_env.yml` → `conda activate hier_env` |
| `environments/entitymatch.yml` | Magellan | 3.9.21 | `conda env create -f environments/entitymatch.yml` |
| `environments/blocking_env.yml` / `blocking_env2.yml` | blocking benchmark | 3.10 / 3.11 | `conda env create -f environments/blocking_env.yml` |
| `environments/requirements.txt` | WordCooc, Magellan, GPT, data gen (venv) | 3.10 | `python -m venv .venv && source .venv/bin/activate && pip install -r environments/requirements.txt` |

Key pins (`requirements.txt`): `torch==2.8.0`, `transformers==4.41.2`, `numpy==1.26.4`,
`pandas==2.3.2`, `scikit-learn==1.6.1`, `xgboost==2.1.4`, `py-entitymatching==0.4.2`,
`codecarbon==3.0.7`, `datasets==4.1.1`, `openai==2.3.0`, `accelerate==0.26.1`.

**GPU:** the neural models were run on a 48 GB GPU partition (`gpu-vram-48gb`), CUDA 12.8
(conda envs pin `torch==2.9.0+cu128`). Ditto needs NLTK stopwords.

> There is a minor version skew: the conda `.yml` files pin `torch==2.9.0+cu128` /
> `transformers==4.49.0` while `requirements.txt` pins `torch==2.8.0` / `transformers==4.41.2`.
> The **conda envs are authoritative** — they are what the SLURM scripts activate.

---

## 3. Data files and naming convention

### Filename pattern

```
products{CC}cc{RR}rnd{UUU}un_{split}_{size}[.json.gz | .pkl.gz | .txt]
        └──corner-case──┘└difficulty┘ └train/valid/gs┘└small/medium/large┘
```

| Token | Values | Meaning |
|---|---|---|
| `{CC}cc{RR}rnd` | `20cc80rnd`, `50cc50rnd`, `80cc20rnd` | corner-case share (cc + rnd = 100) |
| `{UUU}un` | `000un`, `050un`, `100un` | test difficulty: seen / half-seen / unseen |
| `{split}` | `train`, `valid`, `gs` | train / validation / gold-standard (= test) |
| `{size}` | `small`, `medium`, `large` | training size |

### Where the datasets live

| Content | German | English |
|---|---|---|
| Train sets (`.json.gz`) | `data/derived/training-sets/` | `data/derived_en/training-sets/` |
| Validation sets | `data/derived/validation-sets/` | `data/derived_en/validation-sets/` |
| Gold standards (test) | `data/derived/gold-standards_adjusted/` | `data/derived_en/gold-standards_adjusted/` |
| Preprocessed pickles | `data/processed/…` (`preprocessed_` prefix) | `data/processed_en/…` |

### Train / valid / test triple (example: variant `80cc20rnd · large · unseen`)

```
train : data/processed/training-sets/preprocessed_products80cc20rnd000un_train_large.pkl.gz
valid : data/processed/validation-sets/preprocessed_products80cc20rnd000un_valid_large.pkl.gz
test  : data/processed/gold-standards_adjusted/preprocessed_products80cc20rnd100un_gs.pkl.gz
```

> The `productsmulti…` family in these folders is a separate experimental variant and is
> **explicitly skipped** by every model-prep script (`if 'multi' in file: continue`). It is *not*
> one of the 27 benchmark variants.

### Per-model formatted data

| Model | Format | German location |
|---|---|---|
| WordCooc | BoW co-occurrence sparse vectors `.pkl.gz` | `data/processed/wordcooc/learning-curve_adjusted/` |
| Magellan | split tables + `formatted/*.csv` + `.metadata` | `data/processed/magellan/learning-curve_adjusted/` |
| Ditto / HierGAT | serialized `COL … VAL …` text, one pair per line | `data/processed/ditto/data/final_output/`, `data/processed/hiergat/data/final_output/` |
| R-SupCon / RoBERTa | consumes the raw preprocessed pickles directly | `data/processed/training-sets/`, `data/processed/gold-standards_adjusted/`; pretrain data `data/processed/pre-train/{category}/` |

English equivalents are under `data/processed_en/…`.

### Data-preparation / formatting code

- `src/processing/prepare-data.py`, `prepare-data_en.py` — build derived sets / pretrain id selection
- `src/processing/new_testset_all_preprocessings.py` — regenerate all preprocessings for the test sets
- `src/processing/process-to-wordcooc.py` — WordCooc formatting
- `src/processing/process_to_magellan.py` — Magellan formatting
- `src/processing/preprocess_data_for_ditto_hiergat.py` — Ditto/HierGAT serialization
- `src/processing/preprocessing_data.py`, `preprocessing_data_english.py` — base preprocessing
- Dataset generation entry point: `slurm_runs/generate-sets-final.sh` → `notebooks/USB_code/generate-sets-final.py`

> Caveat: several processing scripts hard-code the `data/processed_en/…` output path. The
> German vs. English split is driven by running the `_en` vs. non-`_en` variant / editing that
> path, not by a CLI flag.

### English translation pipeline

`src/translate_to_english/` — `transalte_datasets_to_english_multiple_batches.py` (OpenAI batch
translation), `repair.py`, `reprocess_incomplete_items.py`, `export_untranslated_ids.py`,
`test_language.py`. Translation cost reports: `data/cost_reports/*.json`.

---

## 4. Configurations, hyperparameters, seeds

Per-model summary (verified from the run scripts and configs):

| Model | Seeds | lr / batch / epochs / max_len | Backbone | Early stopping |
|---|---|---|---|---|
| **WordCooc** | 42 + runs 1,2,3 | `RandomizedSearchCV` n_iter=500, scoring=f1, cv=PredefinedSplit | — (BoW) | — |
| **Magellan** | 42 + runs 1,2,3 | `RandomizedSearchCV` n_iter=500, scoring=f1 | — (Magellan feats) | — |
| **RoBERTa baseline** | 0,1,2 | 5e-5 / 1024 / 50 / — | roberta-base | patience 10 |
| **R-SupCon pretrain** | 42 | 5e-5 / 1024 / 200, temp 0.07, warmup 0.05 | roberta-base | — |
| **R-SupCon finetune** | 0,1,2 | 5e-5 / 64 / 50 | roberta-base + contrastive ckpt | patience 10 |
| **HierGAT** | 0,1,2 | 5e-6 / 16 / 50 / 256, `--split` | roberta-base | patience 10 |
| **Ditto** | 0,1,2 | 5e-5 / 64 / 50 / 256, `--da del` | roberta-base | none (50 epochs, best-dev-F1) |
| **GPT-5.2** | — | zero-shot (API) | gpt-5.2 | — |

Config / hyperparameter source files:

| Model | Config / hyperparameters | Training entry point |
|---|---|---|
| Ditto | `src/models/ditto/configs.json` (DE), `configs_en.json` (EN); loop in `all_runs.py` / `all_runs_de.py` | `src/models/ditto/train_ditto.py`, `train_ditto_english.py`; core `ditto_light/ditto.py` |
| HierGAT | `src/models/hiergat/task.json`, `task_en.json`; loops in `all_runs.py` / `all_runs_de.py` (+ split runners in `runs/`) | `src/models/hiergat/train.py`, `train_en.py` |
| WordCooc | grids hard-coded in `run_wordcooc.py` | `src/models/wordcooc/run_wordcooc.py` (+ `_codecarbon[_english].py`) |
| Magellan | grids hard-coded in `run_magellan.py` | `src/models/magellan/run_magellan.py` (+ `_with_codeCarbon[_english].py`) |
| R-SupCon | flags in `run_pretraining.sh` + `run_finetune_siamese.sh` (+ `_en`) | `run_pretraining.py`, `run_finetune_siamese.py`, `run_finetune_siamese_efficiency_tracker.py` |
| RoBERTa baseline | flags in `run_confidence_test.sh` (+ `_en`) | `src/models/transformer_bert_confidence/run_finetune_baseline.py`, `_en.py` |

**Important reproduction facts**

- Ditto data augmentation: `da=del` (span deletion), `dk=None` (no domain knowledge),
  `summarize=False`, `alpha_aug=0.8`.
- WordCooc & Magellan hyperparameter search: `RandomizedSearchCV(n_iter=500, scoring='f1',
  cv=PredefinedSplit, n_jobs=4)` over 6 classifiers (NaiveBayes, XGBoost, RandomForest,
  DecisionTree, LinearSVC, LogisticRegression). Feature combo `brand+name+desc+price`.
- Backbone: although `bert-base-german-cased` appears as a **commented** option in HierGAT,
  R-SupCon, and the RoBERTa baseline, **every committed run uses `roberta-base`** — including the
  German runs. No German-specific backbone is active.
- Checkpoint saving is mostly disabled in the committed scripts (`--save_model` not passed / model
  save commented out); the deep models keep the best-dev-F1 model in memory and write metrics only.
  R-SupCon/RoBERTa keep the best checkpoint per seed (`save_total_limit=1`,
  `load_best_model_at_end`).
- Deviation from the original WDC-Products / ditto code: `da=del` fixed augmentation, no
  domain-knowledge injection, evaluation replicated across the three unseen gold standards, and
  CodeCarbon energy tracking wrapped around training (`OfflineEmissionsTracker`,
  `country_iso_code="DEU"`, €0.30/kWh).
- Exception: `src/models/ditto/all_runs_de.py` was hand-edited to `seeds [3,4]` for a single German
  re-run of `large / 80cc20rnd` — adjust back to `range(3)` for a full reproduction.

---

## 5. Runnable commands

All models are launched through SLURM wrappers in `slurm_runs/`. Each wrapper loops over the
9 (size × corner-case) configs and 3 seeds.

### Quick single-model example

```bash
# Ditto, one German run: large / 80cc20rnd / seed 0
conda activate ditto_env_gpu
CUDA_VISIBLE_DEVICES=0 python src/models/ditto/train_ditto.py \
  --task final_large_80cc20rnd000un \
  --logdir src/models/ditto/results/ \
  --run_id 0 --batch_size 64 --max_len 256 --lr 5e-5 \
  --n_epochs 50 --finetuning --lm roberta --da del
```

### GPT-5.2, one variant

```bash
source .venv/bin/activate
python -u src/models/gpt/gpt_batch_german.py --cc="80cc20" --un="100" --gptmodel="gpt-5.2"
```

### Full runs per model (all 9 configs × 3 seeds)

| Model | German | English |
|---|---|---|
| WordCooc | `sbatch slurm_runs/run_wordcooc.sh` | (same script, `_english` variant) |
| Magellan | `sbatch slurm_runs/run_magellan.sh` | (`_english` variant) |
| RoBERTa baseline | `sbatch src/models/transformer_bert_confidence/run_confidence_test.sh` | `…/run_confidence_test_en.sh` |
| R-SupCon | `sbatch src/models/r-supCon/run_pretraining.sh` then `…/run_finetune_siamese.sh` | `…_en.sh` variants |
| HierGAT | `sbatch slurm_runs/hier_de.sh` | `sbatch slurm_runs/run_hiergat.sh` |
| Ditto | `sbatch slurm_runs/run_ditto_de.sh` | `sbatch slurm_runs/run_ditto.sh` |
| GPT-5.2 | `sbatch slurm_runs/gpt_de.sh` | `sbatch slurm_runs/gpt_en.sh` |

Post-hoc prediction adjustment: `slurm_runs/create_roberta_pred.sh`, `create_rsup_pred.sh`.

> To reproduce all 27 variants for a neural model, run its wrapper once (it loops the 9
> size×corner-case configs × 3 seeds internally) — each trained model is then evaluated against
> the `000un/050un/100un` gold standards to cover the 27 variants. For a **full reproduction**,
> run every wrapper in both the DE and EN columns.

---

## 6. GPT-5.2 reproduction

- **Model identifier:** `gpt-5.2` (passed via `--gptmodel`). `gpt-5-mini` and `gpt-4o` also
  supported in the pricing blocks; **gpt-5.2 is the published model**.
- **Access:** OpenAI **Batch API**, endpoint `/v1/chat/completions`, `completion_window="24h"`.
- **Prompt variants:** *easy* (`gpt_batch_{german,english}.py`, tag `easy_prompt`) and *hard/strict*
  (`gpt_batch_{german,english}_new_prompt.py`, tag `hard_prompt`).
- **Scale / cost:** cost is computed dynamically from token usage (no hard-coded total). The batch
  input `.jsonl` files under `data/batch_inputs/` contain ~279k request lines across all
  languages/prompts/models/splits (the thesis' "~160,000" refers to the core matching subset).
  gpt-5.2 pricing used: $0.875 / 1M input, $7.00 / 1M output.

### Code

| File | Purpose |
|---|---|
| `src/models/gpt/gpt_batch_german.py` | DE, easy prompt: build `.jsonl`, submit, poll, parse, F1 |
| `src/models/gpt/gpt_batch_german_new_prompt.py` | DE, hard prompt |
| `src/models/gpt/gpt_batch_english.py` | EN, easy prompt |
| `src/models/gpt/gpt_batch_english_new_prompt.py` | EN, hard prompt |
| `src/models/gpt/results_report__de.py` / `results_report_en.py` | aggregate CSV + energy logs → Excel summary |
| `src/models/gpt/gpt_category_analysis.py` | per-category F1 |
| `src/models/gpt/testset_error_analysis.py` | re-run mismatches via Responses API + web search |
| `src/models/gpt/secret_config.py` | holds `OPENAI_API_KEY` (git-ignored — rotate before release) |

### Prompts (verbatim)

**German — easy** (`gpt_batch_german.py`):
```
Beziehen sich diese beiden Produktbeschreibungen auf dasselbe reale Produkt?
Antworte nur mit Ja oder Nein.
Produkt 1: {e1}
Produkt 2: {e2}
```

**English — easy** (`gpt_batch_english.py`):
```
Do these two product descriptions refer to the same real-world product?
Answer with Yes or No only.
Product 1: {e1}
Product 2: {e2}
```

**German — hard/strict** (`gpt_batch_german_new_prompt.py`):
```
Du bist ein Experte für Produktabgleich. Deine Aufgabe ist zu entscheiden, ob sich zwei
Produktdatensätze auf das EXAKT gleiche Produkt beziehen (gleiche GTIN/SKU).
Analysiere die bereitgestellten Datensätze sorgfältig und gib deine Entscheidung strikt als
Ja oder Nein zurück.

KRITISCH: Produktvarianten sind KEINE Übereinstimmungen. Unterschiedliche Größen, Farben,
Konfigurationen oder Verpackungsmengen sind UNTERSCHIEDLICHE Produkte mit unterschiedlichen GTINs.

Richtlinien:
- Ja NUR, wenn sich die Datensätze auf exakt dasselbe Produkt beziehen, das dieselbe GTIN/denselben Barcode hätte
- Nein, wenn es sich um Varianten derselben Produktlinie handelt (unterschiedliche Größe, Farbe, Kapazität usw.)
- Nein bei widersprüchlichen zentralen Identifikationsmerkmalen (Modellnummern, Abmessungen, Kapazität, Farbe und Konfiguration)
- Fehlende Attribute allein sind KEIN Widerspruch.
- Eine Übereinstimmung erfordert positive Evidenz der Gleichheit
- Antworte AUSSCHLIESSLICH mit Ja oder Nein.

Produkt 1: {e1}
Produkt 2: {e2}
```

**English — hard/strict** (`gpt_batch_english_new_prompt.py`):
```
You are an expert product matcher. Your task is to decide if two product records refer to the
EXACT same product (same GTIN/SKU).
Analyze the provided records carefully and return your decision as strict Yes or No.

CRITICAL: Product variants are NOT matches. Different sizes, colors, configurations, or package
quantities are DIFFERENT products with different GTINs.

Guidelines:
- Yes ONLY if records refer to the exact same product that would have the same GTIN/barcode
- No if they are variants of the same product line (different size, color, capacity, etc.)
- No if conflicting core identifying attributes (model numbers, dimensions, capacity, color, and configuration)
- Missing attributes alone are NOT a conflict.
- A match requires positive evidence of equivalence
- Respond ONLY with Yes or No.

Product 1: {e1}
Product 2: {e2}
```

Entity serialization (`process_record`): space-joined non-null fields —
EN `Brand: … Name: … Price: … Description: …`, DE `Marke: … Name: … Preis: … Beschreibung: …`;
`/` replaced by space; sent as a single `user` message (no system message).

### Yes/No → label parsing (robust)

```python
# German
if "ja" in ans.lower()  or ans.lower() in ("ja", "1"):   answer_int = 1
elif "nein" in ans.lower() or ans.lower() in ("nein","0"):answer_int = 0
else: answer_int = -1          # invalid — dropped before F1
# English: "yes"/"true"/"1" → 1, "no"/"false"/"0" → 0, else -1
match = int(answer_int == meta["label"])
```

### Batch request format & I/O

- Request line: `{"custom_id": pair_id, "method":"POST", "url":"/v1/chat/completions",
  "body":{"model": gptmodel, "messages":[{"role":"user","content": prompt}]}}`; sidecar
  `_meta.json` maps `pair_id → {label, entity_1, entity_2, is_hard_negative}`.
- Lifecycle: `client.files.create(purpose="batch")` → `client.batches.create(...)` → poll
  `retrieve` every 60 s → `client.files.content(output_file_id)`.
- Read-back: answer from `response.body.choices[0].message.content`; usage from
  `body.usage.prompt_tokens / completion_tokens`.
- Locations: inputs `data/batch_inputs/gpt_{de,en}/<model>/…`, results
  `data/batch_results/gpt_{de,en}/<model>/…`.

### API key setup (do NOT commit a key)

```bash
export OPENAI_API_KEY="sk-…"     # keep secret_config.py git-ignored, or switch to os.environ
```

---

## 7. Evaluation & result-table code

Precision / recall / F1 are computed for the **match class**, then averaged over the three seeds.

| Model | Analysis / aggregation script |
|---|---|
| WordCooc | `src/models/wordcooc/results_analysis.py`, `results_analysis_de.py`, `wordcooc_f1.py` |
| Magellan | `src/models/magellan/results_analysis.py`, `results_analysis_en.py` |
| RoBERTa baseline | `src/models/transformer_bert_confidence/analysis_de.py`, `analysis_en.py`; calibration `confidence_evaluation_de.py`, `confidence_evaluation_en.py` |
| R-SupCon | `src/models/r-supCon/analysis_de.py`, `analysis_en.py` |
| HierGAT | `src/models/hiergat/analysis_de.py`, `analysis_en.py` |
| Ditto | `src/models/ditto/analysis_de.py`, `analysis_en.py`, `results_analysis.py` |
| GPT | `src/models/gpt/results_report__de.py`, `results_report_en.py`, `gpt_category_analysis.py` |
| Category / language comparison | `data_analysis/` (`category_distribution.py`, `category_summary.py`, `dataset_analysis.py`, `excel.py`, `confidence.py`); `notebooks/Categories/` |

Each `analysis_*.py` produces a per-model `*_experiment_summary_{de,en}.xlsx` (precision, recall,
F1, accuracy, energy, CO₂, runtime, cost) and a per-category breakdown. The English-minus-German
language comparison and category evaluation are assembled from these summaries.

---

## 8. Reference results (expected numbers)

> All result folders below are **git-ignored** — they exist on disk but are not committed.
> These are the FINAL published results (see [§9](#9-old--superseded-do-not-publish) for what to avoid).

| Model | German results | English results |
|---|---|---|
| WordCooc | `src/models/wordcooc/model_output_adjusted_ts/` | `src/models/wordcooc/model_output_en/` |
| Magellan | `src/models/magellan/model_output_adjusted_ts/` | `src/models/magellan/model_output_en/` |
| RoBERTa baseline | `src/models/transformer_bert_confidence/reports/` | `…/reports_en/` |
| RoBERTa calibration study | `…/reports_calibration_de/final_calibration_table.csv` | `…/reports_calibration_en/final_calibration_table.csv` |
| R-SupCon | `src/models/r-supCon/reports/` | `src/models/r-supCon/reports_en/` |
| HierGAT | `src/models/hiergat/output/` | `src/models/hiergat/output_en/` |
| Ditto | `src/models/ditto/output/` | `src/models/ditto/output_en/` |
| GPT-5.2 | `src/models/gpt/reports_de/gpt-5.2/` | `src/models/gpt/reports_en/gpt-5.2/` |

**Format of the raw results**

- **WordCooc / Magellan:** per-run CSVs under `…/reports/…/`; per-seed prediction CSVs under
  `…/predictions/`; best thresholds `analysis/best_thresholds_f1.csv`; aggregated Excel
  `analysis/<model>_experiment_summary_de.xlsx`.
- **R-SupCon / RoBERTa:** per-seed folders `…/{0,1,2}/` with HuggingFace `all_results.json` /
  `predict_*_results.json` (F1, precision, recall, accuracy, runtime).
- **HierGAT:** one `.txt` per (size × dataset × seed), e.g.
  `output/final_large_20cc80rnd000un_lr=5e-06_id=0_batch=16_lm=roberta_adjusted.txt`, holding
  `{'best_test_f1', 'best_test_f1_050', 'best_test_f1_100'}`.
- **Ditto:** one `.txt` per run/seed, e.g.
  `output/final_large_20cc80rnd000un_lm=roberta_da=del_dk=None_su=False_id=0_adjusted_testset`,
  holding `{'best_f1', 'best_f1_050', 'best_f1_100'}`. (`results/` / `results_en/` hold only
  TensorBoard event logs — not metrics.)
- **GPT-5.2:** full prediction CSVs under `reports_{de,en}/gpt-5.2/csv_results/`
  (`products_<cc>_<un>un_batched_<lang>_<easy|hard>_prompt.csv`) + per-config F1 text under
  `…/f1/`.

**Aggregated summary tables (best single-file entry points)**

- `src/models/*/…/analysis/<model>_experiment_summary_{de,en}.xlsx`
- GPT: `src/models/gpt/analysis/gpt_experiment_summary_{de,en}_{easy,hard}_prompt.xlsx`
- Calibration: `src/models/transformer_bert_confidence/reports_calibration_{de,en}/final_calibration_table.csv`

**Cost / energy (efficiency)**

- `data/efficiency_tracker_original/` — per-run CodeCarbon CSV+JSON per model
  (`ditto/`, `hiergat/`, `magellan/`, `r_supCon/`, `roberta/`, `wordcooc/`, `gpt_de/`, `gpt_en/`,
  each with `_en` variants) + roll-ups. **Skip `gpt_old/`.**
- `data/cost_reports/*.json` — GPT/translation API cost per dataset variant.

**Category & error analysis**

- Per model: `…/analysis/categories/`
- GPT category: `src/models/gpt/analysis/{de,en}/gpt-5.2/`
- Test-set label corrections / error analysis: `src/models/gpt/error_analysis/`

---

## 9. OLD / superseded (do NOT publish)

- `old_items/` (entire directory)
- `src/models/gpt/reports_old/` (includes `gpt_4o_mini/`)
- `data/efficiency_tracker_original/gpt_old/` and the lowercase `r_supcon/` duplicate
- `data/derived_en_old/`, any `*_old` folders
- Ditto `results/` and `results_en/` (TensorBoard logs only — not the final metrics)

**One thing to verify before publishing:** the HierGAT English summary xlsx is split awkwardly
(`hiergat/output/analysis/hiergat_experiment_summary_de.xlsx` for German, but both
`hiergat/analysis/` and `hiergat/analysis_en/` hold an `_en` xlsx) — confirm which HierGAT English
summary is authoritative.

---

## 10. Publishing checklist

Because `data/` and all `results*/reports*/output*/model_output*/checkpoints*` folders are
git-ignored, decide explicitly what to add:

- [ ] **Rotate the OpenAI key** and change `secret_config.py` → `os.environ["OPENAI_API_KEY"]`.
- [ ] Add the small, high-value result tables (the `*_experiment_summary_*.xlsx`, GPT `f1/` text
      files, `final_calibration_table.csv`, `data/cost_reports/`) to the repo, or attach them as a
      release artifact — they are currently git-ignored.
- [ ] Decide how to distribute the datasets under `data/derived*/` and `data/processed*/`
      (too large for GitHub → external archive / Zenodo, referenced from this file).
- [ ] Publish per-seed raw predictions (GPT `csv_results/`, model `predictions/`) or at minimum the
      per-seed result JSON/txt so numbers can be recomputed.
- [ ] Add a small script that reads the per-seed results, averages over seeds, and diffs against the
      published summary tables (a wrapper around the existing `analysis_*.py`).
- [ ] Remove `.DS_Store` and confirm `old_items/`, `*_old` stay excluded.
- [ ] Point `README.md` / the "How to Use the Benchmark" website chapter at this file.

---

*Generated as an index of the reproduction assets. Every path above was verified to exist on disk
at the time of writing; git-ignored paths are marked as such.*
