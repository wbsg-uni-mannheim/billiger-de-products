# Analysis key: open after annotation

`sampling_key.csv` joins to the three annotation files on `annotation_id` and
contains model judgments. `manifest.json` records source hashes and random seeds.

The sampling population is the 2,999 offers with complete three-model judgments:

| Stratum | Population | Human sample |
| --- | ---: | ---: |
| At least one critical-error flag | 75 | 75 |
| Unanimous error-free | 2,458 | 100 |
| Remaining minor/disagreement cases | 466 | 50 |

The first two samples reuse the existing audit selections. The final 50 are a
uniform random sample from the remaining stratum with seed 20260905. The combined
225 offers are shuffled with seed 20260906 and split into three batches of 75.
The source audit sampled 3,000 test offers. Offer 54158093151 lacks one valid model
judgment and is outside this stratified population. Account for it separately
before reporting an estimate for the complete original audit sample.

Report human critical-error counts and rates by stratum. Estimate the rate in the
2,999-offer population as the sum of each stratum's human rate multiplied by its
population share: (75*r_flagged + 2458*r_clean + 466*r_other) / 2999. The unweighted
rate among the 225 deliberately enriched offers is not a population estimate.
Include uncertainty appropriate to this stratified sampling design. The flagged
stratum is a census within this population. Never count unresolved `uncertain`
labels or blank cells as error-free. Adjudicate them or report bounds explicitly.

These are offer-level translation judgments on test offers. They do not directly
estimate changed pair labels or translation quality in train-only offers. A zero
count in 100 clean-stratum samples alone cannot establish a near-zero population
error rate. Report human findings separately from model votes and agreement.

Reproduction: run `python src/translation_audit/prepare_manual_annotation.py` from
the repository root only when the output directory is absent. The generator
refuses to overwrite any existing annotations and verifies source-text identity,
sample uniqueness, and CSV round trips.
