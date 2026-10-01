# Reproducing the benchmark results

Run all commands from the repository root. The shell files in `slurm_runs/` and `src/models/*/` contain the fixed grids, seeds, and hyperparameters. They can be submitted with `sbatch` or run with `bash` after adapting the `#SBATCH` resource lines.

## 1. Environment

Python 3.10 and a CUDA GPU are recommended for the neural matchers.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements.txt
python -m nltk.downloader punkt stopwords
```

Captured Conda environments: `environments/entitymatch.yml` (Magellan, needs `py-entitymatching`), `environments/ditto_env_gpu.yml` (Ditto), `environments/hier_env.yml` (HierGAT). Launchers call `python`; set `BILLIGER_ENV` to an environment prefix to put it on `PATH` (`slurm_runs/env.sh`), and `ENTITYMATCH_ENV` for the Magellan jobs.

## 2. Data

The released pairs are in `data/solute_de` and `data/solute_en` (see [README.md](README.md) for the format). Within each corner-case family, no offer and no pair occurs in more than one split type. Version 1.1 changes and the label provenance of every changed pair are documented in [docs/CHANGELOG.md](docs/CHANGELOG.md) and `docs/version-1.1/`. The v1.0 split cleaning is documented in `reports/split_cleaning/`.

Model selection for all reference results uses the Seen validation set of the same family and size (`*000un_valid_<size>`). The `050un` validation files contain 50 % unseen products and are released as an alternative for Half-Seen model selection.

## 3. Preprocessing

```bash
python src/processing/prepare_pairs.py
python src/processing/prepare_pretraining.py
python src/processing/prepare_ditto_hiergat.py
python src/processing/prepare_wordcooc.py
python src/processing/prepare_magellan.py   # in the entitymatch environment
```

Generated inputs go to `data/processed/` and `data/processed_en/` (ignored by Git). `slurm_runs/prepare_classical.sh` runs the two classical preparation steps on the cluster.

## 4. Main experiments (27 variants per language)

Supervised matchers train one model per corner-case share and size and evaluate it on the Seen, Half-Seen, and Unseen test sets. Neural matchers use seeds 0, 1, 2; WordCooc and Magellan repeat their training three times and select the classifier on validation. Outputs go to `results/generated/` (ignored by Git).

```bash
# classical matchers
bash slurm_runs/run_wordcooc.sh
bash slurm_runs/run_magellan.sh

# RoBERTa (batch size 32, learning rate 5e-5)
bash src/models/transformer_bert_confidence/run_confidence_test.sh
bash src/models/transformer_bert_confidence/run_confidence_test_en.sh

# XLM-R (80 % corner cases, learning rate 2e-5, warmup 0.1, early-stopping patience 25)
bash slurm_runs/xlmr_main_de.sh
bash slurm_runs/xlmr_main_en.sh

# R-SupCon: pre-train, then fine-tune
bash src/models/r-supCon/run_pretraining.sh
bash src/models/r-supCon/run_pretraining_en.sh
bash src/models/r-supCon/run_finetune_siamese.sh
bash src/models/r-supCon/run_finetune_siamese_en.sh

# Ditto and HierGAT (roberta-base backbone)
bash slurm_runs/run_ditto_de.sh
bash slurm_runs/run_ditto.sh
bash slurm_runs/hier_de.sh
bash slurm_runs/run_hiergat.sh
```

GPT-5.2 (`gpt-5.2-2025-12-11`, Batch API, charges apply) runs both prompts on all 18 test sets. The key is read from the environment only:

```bash
export OPENAI_API_KEY=...
bash slurm_runs/gpt_de.sh
bash slurm_runs/gpt_en.sh
```

## 5. Cross-language experiment

All supervised matchers train on the German `80cc20rnd000un` large training set, select on the German `80cc20rnd000un` large validation set (`slurm_runs/cross_language_protocol.sh`), and are evaluated on the five language variants of the 80 % Half-Seen test set (4,427 pairs each). The Mixed variant (`random` in file names) assigns about a quarter of the pairs to each language combination, stratified by label and hard-negative status (seed 42). Pair ids, labels, and the seen/unseen composition are identical in all variants. GPT-5.2 receives the German prompt in every variant and is repeated three times. The XLM-R cross-language training uses the same data, settings, and seeds as its main-grid 80 % large cell and is deterministic, so its DE-DE scores equal that cell.

```bash
sbatch slurm_runs/cross_language_prepare.sh
# after it has finished
sbatch slurm_runs/cross_language_wordcooc.sh
sbatch slurm_runs/cross_language_magellan.sh
sbatch slurm_runs/cross_language_roberta.sh
sbatch slurm_runs/cross_language_rsupcon.sh
sbatch slurm_runs/cross_language_ditto.sh
sbatch slurm_runs/cross_language_hiergat.sh
sbatch slurm_runs/cross_language_xlmr.sh
sbatch slurm_runs/cross_language_gpt.sh
# when all jobs are done
sbatch slurm_runs/cross_language_summarize.sh
```

## 6. Reported results

The numbers in the paper are in `reports/final_results/` (per seed, cell summary, collapsed runs, category analysis); see its [README](reports/final_results/README.md). They were produced by evaluating all matchers on version 1.1 with the scripts in `src/v11/`: RoBERTa, R-SupCon, and XLM-R predict with their selected checkpoints (`infer_transformer.py`), WordCooc and Magellan refit with the hyperparameters selected on validation (`run_classical.py`), Ditto and HierGAT are retrained with the launchers above, and GPT-5.2 reuses identical requests and queries changed pairs (`run_gpt.py`). `src/v11/export_comparison.py` collects the per-run scores and `reports/final_results/build.py` builds the result files.

Collapsed runs: a Ditto or HierGAT run whose selected checkpoint scores less than 0.10 F1 above the all-positive baseline on Seen validation is restarted once with a new seed (`src/v11/restart_campaign.py`, `src/v11/validate_restart.py`) or, if no restart is available, excluded. `reports/final_results/collapsed_runs.csv` lists all such runs.

The category analysis (`reports/final_results/categories/category_f1.py`) uses the per-offer categories in `offer_categories.csv` and scores same-category pairs on the German test sets.

## 7. Audits

- Label audit of 100 test pairs: `reports/label_audit/`.
- Translation audit (three LLMs on 3,000 test offers, human review of 225): `src/translation_audit/`, `reports/translation_audit/`; the human results are summarized by `python src/translation_audit/summarize_human_audit.py`.

## What is not released

- The raw billiger.de offer corpus and the intermediate construction files. The construction code in `construction/` documents the procedure but cannot be run without them.
- Model checkpoints and raw per-pair predictions.
- OpenAI batch request and response files.
