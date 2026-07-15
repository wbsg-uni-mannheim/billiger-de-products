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

### Recommended set and citation of results

If only one benchmark configuration is used, we recommend `80cc20rnd000un`: 80% corner cases and a fully seen test set. The German and English test files are `data/solute_de/gold-standards_adjusted/products80cc20rnd000un_gs.json.gz` and `data/solute_en/gold-standards_adjusted/products80cc20rnd000un_gs.json.gz`. For supervised matchers, use the corresponding `80cc20rnd000un` training and validation files and state the selected training size.

Report results with enough information to identify the exact experiment, for example:

> Billiger.de Products (German), 80% corner cases, seen (`000un`), large training set, F1 = X (mean ± standard deviation over seeds 0, 1, and 2).

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

## 5. Collect the metrics

Create one machine-readable table from all generated scalar output files:

```bash
python src/summarize_results.py
```

The command writes `results/generated/metrics.csv` with the model, language, metric name, optional classical classifier, F1 score, and source file. Compact outputs from the original runs remain under the corresponding `src/models/*/{reports,output,model_output*}` directories and serve as the reference values shown on the website.

## Reproduction boundary

The main benchmark tables are reproducible from the released pairs and commands above. The category-level tables are retained as reference artifacts, but exact category-level regeneration additionally requires the per-offer product-category mapping used in the original analysis. That mapping is not part of the released benchmark rows. This limitation does not affect the overall F1 results.
