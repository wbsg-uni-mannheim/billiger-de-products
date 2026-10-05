# Reference results (benchmark v1.1)

Precision, recall, and F1 of the match class for every matcher, on the v1.1 test sets in `data/solute_de` and `data/solute_en`. These are the numbers reported in the paper.

| File | Content |
|---|---|
| `<model>.csv` | One row per model, table (`main_grid` or `cross_language`), language, corner-case share (`cc`), development set size, test condition, and seed. |
| `summary.csv` | Cell means of precision, recall, and F1, the F1 standard deviation across seeds (`f1_std`), and the number of seeds. |
| `collapsed_runs.csv` | Training runs treated as collapsed, with their validation scores and how they were handled. |
| `build.py` | Script that produces these files from the per-run outputs. |
| `categories/` | Same-category F1 on the German (`category_f1.csv`) and English (`category_f1_en.csv`) test sets, per-run values, record counts, and the category of every offer (`offer_categories.csv`); produced by `categories/category_f1.py`. |

Test conditions in `main_grid`: `Seen` (`*000un_gs`), `Half-Seen` (`*050un_gs`), `Unseen` (`*100un_gs`). Test conditions in `cross_language`: the five language variants of the 80 % corner-case Half-Seen test set (`de_de`, `de_en`, `en_de`, `en_en`, `random` = Mixed), with models trained on German `large` data.

## Seeds and runs

- Supervised matchers: three training seeds per cell. GPT-5.2: one run per main-grid cell and three repetitions per cross-language variant.
- RoBERTa, R-SupCon, and XLM-R reuse their selected checkpoints and predict on the v1.1 test sets. XLM-R's cross-language training repeats its German 80 % `large` main-grid training with the same data, settings, and seeds; it is deterministic, so the DE-DE scores equal that cell. Ditto and HierGAT were retrained with seeds 0, 1, 2. WordCooc and Magellan refit their classifiers with the hyperparameters selected on validation. GPT-5.2 (`gpt-5.2-2025-12-11`) was queried again for pairs that changed in v1.1.
- Model selection uses the Seen validation set (`*000un_valid_<size>`) for every matcher.

## Collapsed runs

A Ditto or HierGAT run counts as collapsed if the Seen-validation F1 of its selected checkpoint is less than 0.10 above the F1 of predicting every pair as a match. The decision uses validation data only. A collapsed run is replaced by one restart with a new seed if that restart passes the same rule, otherwise it is excluded. `collapsed_runs.csv` lists all ten collapsed runs: five were replaced, five were excluded, so twelve main-grid cells (Ditto DE 80 % medium and large, HierGAT DE 50 % small and 80 % small) average fewer than three seeds (`seeds` column in `summary.csv`).
