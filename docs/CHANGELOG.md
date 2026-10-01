# Changelog

## Version 1.1 (September 2026)

Version 1.1 makes the splits fully disjoint at the offer level. It is the version described in the paper and the one in `data/solute_de`, `data/solute_en`, and the zip files. Version 1.0 remains available in the git history (tag `v1.0.0`).

All changes apply identically to German and English.

**Unchanged.** All training sets, all Seen validation sets (`*000un_valid_*`), and all Unseen test sets (`*100un_gs`) are byte-identical to version 1.0 (21 of 45 files per language).

**Changed files.** The Seen test sets (`*000un_gs`), the Half-Seen test sets (`*050un_gs`), and the `050un`/`100un` validation sets (24 files per language).

1. **Removed overlapping test pairs.** In version 1.0, 102 test pairs contained an offer that also occurred in a training or Seen validation set. These pairs were removed (92 from Seen, 10 from Half-Seen test sets). List: `docs/version-1.1/removed_test_pairs.csv`.
2. **Half-Seen offers aligned with Seen.** In the Half-Seen test and validation sets, the offers of seen products now are the offers reserved for the corresponding Seen test or validation set. Offers of unseen products are unchanged. This removes offer overlap between Half-Seen sets and training data.
3. **Labels of new pairs.** Pairs created by step 2 received labels as follows (`docs/version-1.1/label_decisions.jsonl`, 18,507 decisions):
   - validation pairs: same source product means match (12,296 pairs), as for all validation sets in version 1.0;
   - test pairs between different source products without a known audited match: non-match (6,147 pairs);
   - 15 remaining test pairs: labelled by a model-based text check with high confidence (3 matches, 12 non-matches);
   - 49 test pairs per language whose identity could not be established were replaced by unused pairs from the audited version 1.0 test sets with their original labels (18 matches, 31 non-matches; `docs/version-1.1/replacement_pairs.csv`, `replacement_plan_49.json`).
4. **Checks.** No offer or pair occurs in more than one split type within a family, no open labels, no duplicate or self pairs, and German and English files are aligned in pair ids, order, and labels (`docs/version-1.1/structural_checks.json`, `independent_verification.json`). Pair, offer, product, and label counts per file before and after: `docs/version-1.1/file_statistics.csv`.

The `is_hard_negative` flag is sampling metadata. Replacement pairs keep the flag of the pair they replace.

Resulting test set sizes (pairs, per language): Seen 4,445 / 4,412 / 4,372 and Half-Seen 4,429 / 4,418 / 4,427 for 20 % / 50 % / 80 % corner cases. The cross-language test sets (`data/solute_cross_language.zip`) are built from the version 1.1 Half-Seen 80 % set (4,427 pairs).

Code: `src/processing/prepare_v11_draft.py` (offer alignment and removal) and `src/processing/finalize_v11.py` (label decisions and replacements).

## Version 1.0 (July 2026)

First public release.
