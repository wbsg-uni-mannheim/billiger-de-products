You are auditing a machine translation of a German e-commerce product offer into English. The translation is used in an entity matching benchmark, where two offers must be judged as the same product or not. Your task is to check whether the English version preserves every piece of information that identifies the product, not whether the English is fluent.

Identity-relevant information includes: brand and model names, model and article numbers, variant attributes (size, colour, material, capacity, configuration, gender, age group, package quantity, edition), and all numbers with their units.

Compare the German original and the English translation below and answer with a single JSON object, nothing else, with exactly these keys:

{
  "codes_preserved": "yes" | "no" | "n/a",
  "variants_preserved": "yes" | "no" | "n/a",
  "quantities_preserved": "yes" | "no" | "n/a",
  "omission_or_addition": "none" | "minor" | "critical",
  "could_change_match_decision": "no" | "possibly" | "yes",
  "verdict": "error-free" | "minor error" | "critical error",
  "issue": "<short description of the problem, or empty string>"
}

Definitions:
- "critical error": the translation changes, drops, or adds an identity-relevant attribute so that a match or non-match decision could be different. Examples: a changed model number, a dropped size or colour, a different capacity, a wrong quantity, a wrong number or unit.
- "minor error": awkward or unnatural wording, a mistranslated non-identifying word, changed word order, or a dropped marketing phrase. All identity-relevant information is intact.
- "error-free": the translation preserves all identity-relevant information and reads correctly.
- Use "n/a" when the German text contains no information of that kind.
- Untranslated brand names, model names, and codes are correct, not errors.
- Locale conventions stay unchanged in the benchmark: prices remain in Euro and measurements metric. Do not flag them.

German original
Brand: {brand}
Name: {de_name}
Description: {de_desc}

English translation
Brand: {brand}
Name: {en_name}
Description: {en_desc}
