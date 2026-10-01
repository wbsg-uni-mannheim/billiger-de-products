# Billiger.de Products: A Bilingual Entity Matching Benchmark

Billiger.de Products is an entity matching benchmark for German product offers, with an aligned English translation of every offer. It contains 13,568 offers describing 2,168 products from the German price comparison portal billiger.de, assigned to 13 top-level product categories.

The benchmark varies three dimensions: the share of corner cases (20 %, 50 %, 80 %), the share of products unseen during training (Seen, Half-Seen, Unseen), and the development set size (small, medium, large). German and English files have identical splits, labels, and identifiers. The repository contains the released pairs, the construction code, preprocessing, eight matchers (WordCooc, Magellan, RoBERTa, XLM-R, R-SupCon, HierGAT, Ditto, GPT-5.2 zero-shot), and per-seed reference results.

The current release is version 1.1 (see [docs/CHANGELOG.md](docs/CHANGELOG.md)). The data is released with the permission of solute GmbH, the operator of billiger.de. The records contain no information identifying shops, sellers, or persons.

## Data

| Path | Content |
|---|---|
| `data/solute_de/`, `data/solute_de.zip` | German pairs: `training-sets/`, `validation-sets/`, `gold-standards_adjusted/` (test sets) |
| `data/solute_en/`, `data/solute_en.zip` | English pairs, same files and pair ids |
| `data/solute_cross_language.zip` | The 80 % corner-case Half-Seen test set in five language combinations (DE-DE, DE-EN, EN-DE, EN-EN, Mixed) |

Each gzip-compressed JSON-lines row holds both offers (`id`, `brand`, `name`, `desc`, `price`, `product_id` with suffixes `_left` and `_right`), the `pair_id`, the binary `label`, and `is_hard_negative`. File names encode the corner-case share (`20cc80`, `50cc50`, `80cc20`), the share of unseen products (`000un` Seen, `050un` Half-Seen, `100un` Unseen), and the size of training and validation sets (`small`, `medium`, `large`).

## Recommended configuration and reporting

For a single evaluation we recommend 80 % corner cases, Half-Seen: test on `products80cc20rnd050un_gs.json.gz`, train on `products80cc20rnd000un_train_<size>.json.gz`, and select models on `products80cc20rnd000un_valid_<size>.json.gz` (the Seen validation set used for all reference results). Report, for example:

> Billiger.de Products v1.1 (German), 80 % corner cases, Half-Seen, large training set, F1 = X (mean and standard deviation over three seeds).

Zero-shot methods omit the training size and state the model and prompt.

## Results

Per-seed precision, recall, and F1 of all matchers on all 27 variants of both languages and on the cross-language test sets are in [reports/final_results/](reports/final_results/) (cell means and standard deviations in `summary.csv`, the category-level analysis in `categories/`). The GPT-5.2 prompts in English and German are on the [benchmark website](https://wbsg-uni-mannheim.github.io/billiger-de-products/).

## Reproduction

See [REPRODUCTION.md](REPRODUCTION.md) for environments, preprocessing, the commands for every matcher and experiment, how the reported tables are built, and what is not released. The code that built the benchmark from the billiger.de corpus is in [construction/](construction/).

## Repository layout

```text
data/                 released pairs (v1.1)
docs/                 changelog, version 1.1 label provenance, construction notes, category taxonomy
construction/         benchmark construction code (cleansing, selection, pairs, translation, categories)
src/processing/       model-input preparation and version 1.1 generation
src/models/           matcher implementations
src/cross_language/   cross-language experiment helpers
src/translation_audit/  LLM-based and human translation audit
src/v11/              evaluation of all matchers on version 1.1 (cluster jobs)
slurm_runs/           experiment launchers
reports/              reference results, label audit, translation audit, split cleaning
environments/         Python requirements and Conda environments
website/              benchmark website
```

## Citation

```bibtex
@unpublished{steiner2026billiger,
  author = {Steiner, Aaron and Elagin, Ksenia and Peeters, Ralph and Knopp, Johannes and Bizer, Christian},
  title  = {Billiger.de Products: A Bilingual Entity Matching Benchmark},
  note   = {Submitted to BTW 2027},
  year   = {2026}
}
```
