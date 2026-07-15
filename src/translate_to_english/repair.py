import re
import json
import gzip
import os

import pandas as pd

INVALID_UNICODE_RE = re.compile(r'\\u(?![0-9a-fA-F]{4})')

def repair_json_string(s: str) -> str:
    """
    Attempt safe, mechanical repairs on near-JSON output.
    """

    # Remove invalid \u escapes (replace with Unicode replacement char)
    s = INVALID_UNICODE_RE.sub("�", s)

    # Fix over-escaped newlines
    s = s.replace('\\\\n', '\\n')

    # Remove backslash before non-escape characters
    s = re.sub(r'\\([^"\\/bfnrtu])', r'\1', s)

    s = s.replace("\r", "").replace("\n", "\\n")

    # Trim anything before first { and after last }
    if "{" in s and "}" in s:
        s = s[s.find("{"):s.rfind("}") + 1]

    return s

def load_recovered_translations(error_log_path):
    recovered = {}

    with open(error_log_path, "r", encoding="utf-8") as f:
        errors = json.load(f)

    for row in errors:
        pid = row["pair_id"]
        pid = int(pid)
        raw = row["raw_output"]

        try:
            fixed = repair_json_string(raw)
            parsed = json.loads(fixed)

            required = {"name", "desc"}
            if not required.issubset(parsed):
                continue

            recovered[pid] = {
                "name": parsed["name"],
                "desc": parsed["desc"]
            }

        except Exception as e:
            print(f"Failed to parse output for pair_id {pid}: {e}")
            continue

    return recovered


def salvage_failed_rows(recovered_translations, translation_path):
    patched = 0
    total = 0

    # Read the existing translation file
    #with gzip.open(translation_path, "rt", encoding="utf-8", errors="replace") as fin:
        #rows = [json.loads(line) for line in fin]
    rows = []
    with gzip.open(translation_path, "rt", encoding="utf-8", errors="replace") as fin:
        for i, line in enumerate(fin):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{translation_path} → Bad JSON line {i}: {e}")
                print(f"Line content: {line[:100]}...")  # Print the beginning of the bad line for debugging


    # Update rows based on recovered translations
    for row in rows:
        total += 1
        pid_left = row.get("id_left")
        pid_right = row.get("id_right")

        if pid_left in recovered_translations:
            row["name_left"] = recovered_translations[pid_left]["name"]
            row["desc_left"] = recovered_translations[pid_left]["desc"]
            patched += 1

        if pid_right in recovered_translations:
            row["name_right"] = recovered_translations[pid_right]["name"]
            row["desc_right"] = recovered_translations[pid_right]["desc"]
            patched += 1

    # Write the updated rows back to the translation file
    with gzip.open(translation_path, "wt", encoding="utf-8", errors="replace") as fout:
        for row in rows:
            try:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to write row: {row}, error: {e}")

    return patched, total


if __name__ == "__main__":
    # Example usage
    folder = "training-sets"
    #split = "50cc50"

    error_log = f"data/batch_results/translation_errors/combined__{folder}_errors.json"
    output_base_path = f"data/derived_en/{folder}"

    datasets = [
        f for f in os.listdir(output_base_path)
        if "multi" not in f.lower()
    ]

    recovered = load_recovered_translations(error_log)
    print(f"Recovered translations for {len(recovered)} product IDs")

    for dataset in datasets:
        output_data = os.path.join(output_base_path, dataset)

        patched, total = salvage_failed_rows(
            recovered,
            output_data
        )

        print(f"{dataset}: patched {patched} fields over {total} rows")