# Blind label audit of the released test sets

Residual-noise audit reported in the paper (Section "Test Set Verification").

## Procedure

- Population: all nine German test sets in `data/solute_de/gold-standards_adjusted/`
  (39,959 rows, 38,561 unique pairs after deduplication across configurations,
  positive rate 7.8%, class imbalance 1:11.8).
- Sample: 100 pairs, stratified by released label (50 matches, 50 non-matches),
  fixed seed, shuffled. Drawn by `sample_pairs.py`; the released label is stored
  only in `sample_key.csv` and was not visible during labeling.
- Labeling: one author labeled all 100 pairs blind to the released labels in
  `label_100_pairs.html` (side-by-side records with token diff), using the
  four-rule identity definition from the paper. A second pass reviewed all
  initial disagreements.
- Scoring: `score.py <labels.csv>` prints per-class disagreement with Wilson
  95% intervals.

## Result (2026-08-21, annotator 1 = Aaron Steiner)

- Matches: 50/50 agreement.
- Non-matches: 49/50 agreement. The single disagreement is pair
  4622655287#5006573365 (Triumph Minimizer-BH vs. TRIUMPH & DISASTER Schalen-BH),
  released label 0, annotator label 1.
- Overall: 99/100 agreement, 1% raw disagreement, estimated 1.8% of test pairs
  when weighted by the 1:11.8 class imbalance.

Final labels: `labels_annotator1.json` (pair_id -> label).

The sample was drawn from the version 1.0 test sets (git tag `v1.0.0`); `sample_pairs.py` reproduces it only on those files. In version 1.1, 78 of the 100 pairs are still in the test sets, all with unchanged labels. Weighted by the version 1.1 class imbalance of the unique test pairs (1:12.8), the estimated disagreement rate is 1.9 %.
