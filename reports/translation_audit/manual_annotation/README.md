# Manual translation annotation

Label all three `translations_01.csv` to `translations_03.csv` files, in any order.
Each contains 75 different offers, for 225 offers in total. The offers are shuffled
and all model judgments have been removed.

Fill only `human_verdict` and `human_note`. Compare the German name and description
with their English translations. Judge whether the translation preserves the
information needed to identify the product and distinguish its variants.

| human_verdict | Meaning |
| --- | --- |
| `error-free` | Faithful translation. Natural paraphrases and stylistic differences are acceptable. |
| `minor error` | A translation error that does not change product identity or variant-defining information. |
| `critical error` | A mistranslation, omission, or invented detail that could change a matching decision, such as model, size, material, color, capacity, quantity, or compatibility. |
| `uncertain` | Insufficient evidence or unclear source text prevents a confident judgment. |

Use the most severe error across name and description. For errors or uncertain
cases, quote the relevant German/English fragments in `human_note` and explain
the difference. A note is optional for `error-free`. Judge translation fidelity:
an inconsistency already present in the German source is not itself a translation
error if faithfully retained in English.

Keep the IDs, original text, and column names unchanged. Save as comma-delimited
UTF-8 CSV. Multiline descriptions are correctly quoted CSV fields. In Excel,
import via Data > From Text/CSV and treat all columns as text. Widen the text
columns and enable wrapping for reading. CSV files cannot preserve formatting or
dropdown validation.

Do not consult `analysis_only/` or the earlier model-audit files while labeling.
They contain the selection key and model judgments for analysis after annotation.
If you have already inspected a particular offer's model judgment, record that
in `human_note`. A second annotator should work from a separate blank copy without
seeing your labels, followed by adjudication of disagreements.

The original audit CSVs are preserved. Please label these new files rather than
the earlier files that expose model judgments.
