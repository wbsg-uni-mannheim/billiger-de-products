# Human translation audit: results

Produced by `python src/translation_audit/summarize_human_audit.py` from the three `translations_*.csv` files and `sampling_key.csv`.

| Sampling group | Population | Annotated | Acceptable | Minor | Matching-relevant |
| --- | ---: | ---: | ---: | ---: | ---: |
| At least one model flags a critical error | 75 | 75 | 60 | 4 | 11 |
| All three models judge error-free | 2,458 | 100 | 100 | 0 | 0 |
| Other minor or disagreement cases | 466 | 50 | 47 | 2 | 1 |

Population-weighted matching-relevant error rate: (75 x 11/75 + 2,458 x 0/100 + 466 x 1/50) / 2,999 = 0.68 %.

Labels: `Ja okay` = acceptable, `Nein nicht relevant` = error irrelevant to matching, `Nein relevant` = potentially matching-relevant error. One annotator, blind to the model judgments.
