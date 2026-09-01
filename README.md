# Billiger.de Products: A Bilingual Entity Matching Benchmark

Billiger.de Products is a bilingual entity matching benchmark for matching German product offers and aligned English translations of the same offers. It contains 13,730 offers describing 2,168 products from the German price-comparison platform billiger.de.

The benchmark varies three dimensions: training-set size, the share of difficult corner cases, and the share of products unseen during training. German and English files have identical splits, labels, and identifiers. The repository includes the released pairs, preprocessing code, six supervised matchers, GPT-5.2 zero-shot experiments, and compact reference results.

## Quick start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements.txt

python src/processing/prepare_pairs.py
python src/processing/prepare_pretraining.py
python src/processing/prepare_ditto_hiergat.py
python src/processing/prepare_wordcooc.py
python src/processing/prepare_magellan.py
```

See [REPRODUCTION.md](REPRODUCTION.md) for the complete experiment matrix, model commands, generated-result layout, and the precise reproduction boundary.

## Recommended configuration and result reporting

For a single default evaluation, we recommend `80cc20rnd050un` (80% corner cases, 50% seen products). Use `products80cc20rnd050un_gs.json.gz` as the test set. Supervised matchers should train on the corresponding `80cc20rnd000un` training file, validate on `80cc20rnd050un`, and explicitly state whether the `small`, `medium`, or `large` training split was used.

Report a result in the following form:

> Billiger.de Products (German or English), 80% corner cases, half-seen (`050un`), [training size], F1 = X (mean ± standard deviation over seeds 0, 1, and 2).

Also cite the benchmark paper when available and link to this repository or identify the evaluated repository commit. Zero-shot methods should omit the training size and state the model and prompt used.

## Repository layout

```text
data/solute_de/        released German benchmark pairs
data/solute_en/        released English benchmark pairs
src/processing/        deterministic model-input preparation
src/models/            benchmark implementations and reference metrics
slurm_runs/            complete experiment launchers
environments/          Python and captured Conda environments
website/               benchmark website
```

The benchmark website is in `website/index.html`. The public project URL is [github.com/wbsg-uni-mannheim/billiger-de-products](https://github.com/wbsg-uni-mannheim/billiger-de-products).
