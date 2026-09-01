"""Assert that every derived test file carries the RELEASED gold-standard labels.

The July Ditto/HierGAT runs were scored against a label vector with 100-142 extra
positives per variant. Nothing downstream can detect that on its own, so this gate
runs before any training and aborts the pipeline when a derived file disagrees with
data/solute_{de,en}/gold-standards_adjusted.

    python src/processing/verify_released_serialization.py            # both languages
    python src/processing/verify_released_serialization.py --language de
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# (rows, positives) per configuration and unseen share, from the released files.
EXPECTED = {
    ("20cc80", "000un"): (4454, 407),
    ("20cc80", "050un"): (4429, 352),
    ("20cc80", "100un"): (4449, 360),
    ("50cc50", "000un"): (4427, 364),
    ("50cc50", "050un"): (4418, 346),
    ("50cc50", "100un"): (4458, 369),
    ("80cc20", "000un"): (4440, 342),
    ("80cc20", "050un"): (4437, 361),
    ("80cc20", "100un"): (4447, 348),
}
RELEASED = {"de": Path("data/solute_de"), "en": Path("data/solute_en")}
PROCESSED = {"de": Path("data/processed"), "en": Path("data/processed_en")}
SERIALIZED = ("ditto", "hiergat")


def check(label, rows, positives, expected, failures):
    want_rows, want_pos = expected
    if rows != want_rows or positives != want_pos:
        failures.append(f"{label}: rows={rows} (expected {want_rows}), positives={positives} (expected {want_pos})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()
    languages = ("de", "en") if args.language == "both" else (args.language,)

    failures = []
    checked = 0
    for language in languages:
        for (config, unseen), expected in sorted(EXPECTED.items()):
            name = f"products{config}rnd{unseen}_gs"

            released = pd.read_json(
                RELEASED[language] / "gold-standards_adjusted" / f"{name}.json.gz", lines=True
            )
            check(f"released {language} {name}", len(released), int(released["label"].sum()), expected, failures)
            checked += 1

            pickle_path = PROCESSED[language] / "gold-standards_adjusted" / f"preprocessed_{name}.pkl.gz"
            if pickle_path.exists():
                pairs = pd.read_pickle(pickle_path)
                if check(f"pickle {language} {name}", len(pairs), int(pairs["label"].sum()), expected, failures):
                    # Same rows AND the same label for the same pair, not merely the same totals.
                    merged = released.set_index(released["pair_id"].astype(str))["label"]
                    mine = pairs.set_index(pairs["pair_id"].astype(str))["label"]
                    if not mine.reindex(merged.index).equals(merged):
                        failures.append(f"pickle {language} {name}: per-pair labels differ from released")
                checked += 1
            else:
                failures.append(f"pickle {language} {name}: missing ({pickle_path})")

            for model in SERIALIZED:
                text_path = PROCESSED[language] / model / "data/final_output" / f"preprocessed_{name}.txt"
                if not text_path.exists():
                    failures.append(f"{model} {language} {name}: missing ({text_path})")
                    continue
                lines = [line for line in text_path.read_text(encoding="utf-8").split("\n") if line.strip()]
                positives = sum(1 for line in lines if line.rsplit("\t", 1)[-1] == "1")
                check(f"{model} {language} {name}", len(lines), positives, expected, failures)
                checked += 1

    print(f"Checked {checked} files against the released gold standards.")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)
    print("All derived test files carry the released labels.")


if __name__ == "__main__":
    main()
