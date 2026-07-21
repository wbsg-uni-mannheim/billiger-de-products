# Cross-language evaluation results

Reference results for the cross-language experiment described in
[REPRODUCTION.md](../../REPRODUCTION.md#5-cross-language-experiment).

All supervised matchers are trained on the German DE-DE `80cc20rnd000un` **large**
training split and selected on the German DE-DE `80cc20rnd050un` **large** validation
split. The selected model is then evaluated, without further training, on the five
aligned `80cc20rnd050un` test variants (`de_de`, `de_en`, `en_de`, `en_en`, `random`).
All five variants contain the same 4,437 pairs, pair IDs, labels and seen/unseen
composition; only the language of the left/right record changes. `random` is a
deterministic, label- and hard-negative-stratified mix generated with seed 42.

## Files

| File | Contents |
| --- | --- |
| `metrics.csv` | One row per model / classifier / seed / test variant: precision, recall, F1, source file. |
| `summary.csv` | Mean and standard deviation across runs per model / classifier / test variant. |

The `runs` column in `summary.csv` states how many runs each aggregate is based on.

## Headline F1 (match class)

| Model | runs | DE-DE | DE-EN | EN-DE | EN-EN | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.2 (rule-guided) | 1 | 0.809 | 0.835 | 0.817 | 0.809 | 0.826 |
| GPT-5.2 (simple) | 1 | 0.705 | 0.715 | 0.708 | 0.729 | 0.707 |
| Ditto | 2 | 0.654 | 0.553 | 0.551 | 0.624 | 0.592 |
| RoBERTa | 3 | 0.634 | 0.533 | 0.545 | 0.581 | 0.567 |
| HierGAT | 3 | 0.608 | 0.493 | 0.499 | 0.576 | 0.544 |
| R-SupCon | 3 | 0.485 | 0.470 | 0.465 | 0.423 | 0.452 |
| WordCooc (LogisticRegression) | — | 0.436 | 0.368 | 0.366 | 0.282 | 0.332 |
| Magellan (XGBoost) | — | 0.363 | 0.339 | 0.332 | 0.356 | 0.356 |

Standard deviations are in `summary.csv`; per-seed values are in `metrics.csv`.
The neural matchers aggregate three seeds (`0`, `1`, `2`), except Ditto — see below.

## Excluded run: Ditto seed 0

**The Ditto row aggregates seeds 1 and 2 only. Seed 0 is excluded.**

Ditto seed 0 collapses to the degenerate all-positive solution and predicts every
pair as a match:

```
precision = 0.0813612801442416   (= the positive rate of the test set)
recall    = 1.0
F1        = 0.1504793664026678   (identical for all five test variants)
```

The run was repeated once with identical hyperparameters and produced a
**bit-identical** result, so the collapse is deterministic and reproducible rather
than a transient failure. `train_ditto.py` seeds `random`, `numpy`, `torch` and
`torch.cuda` from `--run_id`, so initialisation and data order are fixed for a given
seed.

Including seed 0 would give Ditto `0.486 ± 0.291` (DE-DE), i.e. a standard deviation
larger than half the mean, which reflects the collapse rather than the matcher's
behaviour. The excluded run is reported here rather than silently dropped; anyone
reproducing the benchmark with seeds 0, 1, 2 will observe it.

## Notes on the other matchers

- **GPT-5.2** is zero-shot; it has no seeds and no training size. Both prompt
  variants were run over all five test variants via the OpenAI Batch API
  (5 x 2 x 4,437 = 44,370 requests). There were **0 unparsable answers**.
- **WordCooc** and **Magellan** repeat their fixed training procedure per test file
  and report one value per classifier; the strongest classifier per family is shown
  above.
- Standard deviations of exactly `0.000` for some classical classifiers indicate a
  deterministic fit for that configuration, not a missing value.
