# Reproducing the benchmark results

This repository contains the released German and English benchmark pairs, preprocessing code, the seven evaluated matchers, fixed experiment commands, and compact reference results. Candidate generation (blocking), translation generation, energy/runtime tracking, calibration experiments, and raw per-pair predictions are intentionally outside this reproduction package.

## 1. Environment

Run commands from the repository root. Python 3.10 and a CUDA-capable Linux machine are recommended for the neural matchers.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environments/requirements.txt
```

`py-entitymatching` can be sensitive to platform and Python versions. A captured environment for Magellan is available in `environments/entitymatch.yml`; captured GPU environments for Ditto and HierGAT are in `environments/ditto_env_gpu.yml` and `environments/hier_env.yml`.

The first use of NLTK-based components may require:

```bash
python -m nltk.downloader punkt stopwords
```

## 2. Released data

The source-of-truth files are already tracked:

| Language | Training and validation pairs | Test pairs |
| --- | --- | --- |
| German | `data/solute_de/{training-sets,validation-sets}` | `data/solute_de/gold-standards_adjusted` |
| English | `data/solute_en/{training-sets,validation-sets}` | `data/solute_en/gold-standards_adjusted` |

Each gzip-compressed JSONL row contains both records, `pair_id`, the binary `label`, and `is_hard_negative`. File names encode the corner-case ratio (`20cc80`, `50cc50`, or `80cc20`), unseen-product share (`000un`, `050un`, or `100un`), and the size (`small`, `medium`, or `large`) on training and validation files.

The released files are pair-disjoint within each corner-case family under the priority `Test > Validation > Train`; offer pairs with conflicting labels were removed. The `000un` validation files contain 100% seen products. Both `050un` and `100un` experiments use a 50%-seen validation split; `100un` refers to the fully unseen test condition, not to model selection. The complete before/after audit is available in `reports/split_cleaning/`.

### Recommended set and citation of results

If only one benchmark configuration is used, we recommend `80cc20rnd050un`: 80% corner cases and 50% seen products. The German and English test files are `data/solute_de/gold-standards_adjusted/products80cc20rnd050un_gs.json.gz` and `data/solute_en/gold-standards_adjusted/products80cc20rnd050un_gs.json.gz`. For supervised matchers, train on the corresponding German or English `80cc20rnd000un` training file and validate on the same language's `80cc20rnd050un` validation file. There is no separate `050un` training file: the unseen-product share is defined for validation and test products relative to the training data. State the selected training size.

Report results with enough information to identify the exact experiment, for example:

> Billiger.de Products (German), 80% corner cases, 50% seen (`050un`), large training set, F1 = X (mean ± standard deviation over seeds 0, 1, and 2).

The result citation should include the benchmark paper once available and this repository or the evaluated commit. For zero-shot methods, replace the training size and seeds with the model name, model version, and prompt variant.

## 3. Preprocessing

Generate all model inputs in this order:

```bash
python src/processing/prepare_pairs.py
python src/processing/prepare_pretraining.py
python src/processing/prepare_ditto_hiergat.py
python src/processing/prepare_wordcooc.py
python src/processing/prepare_magellan.py
```

Generated inputs are written below `data/processed/` and `data/processed_en/`. These directories are ignored by Git because they are deterministically derived from the released JSONL files.

## 4. Run the experiments

The checked-in shell files contain the fixed grids, seeds, and hyperparameters used for the benchmark. They can be submitted with `sbatch` on Slurm or run with `bash` after removing/adapting only the `#SBATCH` resource declarations for the local machine.

### Classical matchers

```bash
bash slurm_runs/run_wordcooc.sh
bash slurm_runs/run_magellan.sh
```

### Neural matchers

```bash
# RoBERTa baseline
bash src/models/transformer_bert_confidence/run_confidence_test.sh
bash src/models/transformer_bert_confidence/run_confidence_test_en.sh

# R-SupCon: pretrain first, then fine-tune
bash src/models/r-supCon/run_pretraining.sh
bash src/models/r-supCon/run_pretraining_en.sh
bash src/models/r-supCon/run_finetune_siamese.sh
bash src/models/r-supCon/run_finetune_siamese_en.sh

# Ditto and HierGAT
bash slurm_runs/run_ditto_de.sh
bash slurm_runs/run_ditto.sh
bash slurm_runs/hier_de.sh
bash slurm_runs/run_hiergat.sh
```

The supervised experiments run seeds 0, 1, and 2 for every combination of language, training size, corner-case ratio, and unseen-product share. Generated checkpoints and metrics are written below `results/generated/` and are ignored by Git.

### GPT-5.2 zero-shot runs

GPT experiments use the OpenAI Batch API and incur API charges. Provide the key only through the environment:

```bash
export OPENAI_API_KEY="..."
bash slurm_runs/gpt_de.sh
bash slurm_runs/gpt_en.sh
```

Both the simple and rule-guided prompts are run on all nine German and nine English test variants. Batch request/response files are stored below the ignored `data/batch_inputs/` and `data/batch_results/` directories; metrics go to `results/generated/gpt/`.

## 5. Cross-language experiment

The cross-language experiment trains every supervised matcher on the German DE-DE `80cc20rnd000un` large training split and selects checkpoints or hyperparameters on the German DE-DE `80cc20rnd050un` large validation split. Training and validation use the same language combination but disjoint pairs, while the validation products remain exactly 50% seen. The selected model is evaluated on the 50% seen test configuration (`80cc20rnd050un`) in five language variants:

| Variant | Left record | Right record |
| --- | --- | --- |
| `de_de` | German | German |
| `de_en` | German | English |
| `en_de` | English | German |
| `en_en` | English | English |
| `random` | German or English | German or English |

`Random-Random` contains an approximately equal number of all four language combinations. Assignment is deterministic, stratified by label and hard-negative status, and generated with seed 42. Pair IDs, labels, identifiers, and the seen/unseen composition remain identical in every variant.

Prepare the shared German inputs and the five aligned test sets:

```bash
sbatch slurm_runs/cross_language_prepare.sh
```

After the preparation job succeeds, submit the model jobs:

```bash
sbatch slurm_runs/cross_language_wordcooc.sh
sbatch slurm_runs/cross_language_roberta.sh
sbatch slurm_runs/cross_language_rsupcon.sh
sbatch slurm_runs/cross_language_ditto.sh
sbatch slurm_runs/cross_language_hiergat.sh
sbatch slurm_runs/cross_language_gpt.sh
```

Magellan requires the `py-entitymatching` environment and has its own preparation step:

```bash
sbatch slurm_runs/cross_language_magellan.sh
```

The GPT job requires `OPENAI_API_KEY` in the submitted environment and incurs Batch API charges. Its German simple and rule-guided prompts are kept fixed across all five test variants so that prompt language does not change between conditions.

The neural matchers run seeds 0, 1, and 2. Each seed trains once on DE-DE, uses only the DE-DE `050un` validation F1 for model selection, and evaluates that selected model on all five test variants. WordCooc and Magellan use the same DE-DE training/validation split and repeat their fixed training procedure for each test file. R-SupCon pre-training uses records from the German training split only; validation records are not included.

After all jobs have completed, collect the results:

```bash
sbatch slurm_runs/cross_language_summarize.sh
```

This writes `results/generated/cross_language/metrics.csv` with model, classifier where applicable, seed, test variant, precision, recall, F1, and the source result file. Mean and standard deviation across seeds are written to `results/generated/cross_language/summary.csv`.

The intended result table is:

| Model | DE-DE | DE-EN | EN-DE | EN-EN | Random-Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| WordCooc |  |  |  |  |  |
| Magellan |  |  |  |  |  |
| RoBERTa |  |  |  |  |  |
| R-SupCon |  |  |  |  |  |
| HierGAT |  |  |  |  |  |
| Ditto |  |  |  |  |  |
| GPT-5.2 Simple |  |  |  |  |  |
| GPT-5.2 Rule-Guided |  |  |  |  |  |

## 6. Collect the benchmark metrics

Create one machine-readable table from all generated scalar output files:

```bash
python src/summarize_results.py
```

The command writes `results/generated/metrics.csv` with the model, language, metric name, optional classical classifier, F1 score, and source file. Compact outputs from the original runs remain under the corresponding `src/models/*/{reports,output,model_output*}` directories and serve as the reference values shown on the website.

## Reproduction boundary

The main benchmark tables are reproducible from the released pairs and commands above. The category-level tables are retained as reference artifacts, but exact category-level regeneration additionally requires the per-offer product-category mapping used in the original analysis. That mapping is not part of the released benchmark rows. This limitation does not affect the overall F1 results.
