"""Build the unique-record inputs used by R-SupCon pre-training."""

import argparse
from pathlib import Path

import pandas as pd


LANGUAGES = {
    "de": Path("data/processed"),
    "en": Path("data/processed_en"),
}
CORNER_CASES = ("20cc80rnd", "50cc50rnd", "80cc20rnd")
SIZES = ("small", "medium", "large")
ATTRIBUTES = ("id", "product_id", "brand", "name", "desc", "price")


def extract_records(pairs):
    sides = []
    for side in ("left", "right"):
        columns = [f"{attribute}_{side}" for attribute in ATTRIBUTES]
        records = pairs[columns].copy()
        records.columns = ATTRIBUTES
        sides.append(records)
    return pd.concat(sides, ignore_index=True).drop_duplicates(subset="id")


def prepare_language(language):
    root = LANGUAGES[language]
    written = 0

    for corner_cases in CORNER_CASES:
        variant = f"products{corner_cases}000un"
        output_dir = root / "pre-train" / variant
        output_dir.mkdir(parents=True, exist_ok=True)

        for size in SIZES:
            train_path = root / "training-sets" / f"preprocessed_{variant}_train_{size}.pkl.gz"
            valid_path = root / "validation-sets" / f"preprocessed_{variant}_valid_{size}.pkl.gz"
            records = pd.concat(
                [
                    extract_records(pd.read_pickle(train_path)),
                    extract_records(pd.read_pickle(valid_path)),
                ],
                ignore_index=True,
            ).drop_duplicates(subset="id")
            records.to_pickle(
                output_dir / f"{variant}_train_{size}.pkl.gz",
                compression="gzip",
            )
            written += 1

    print(f"Prepared {written} {language.upper()} R-SupCon pre-training files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "both" else (args.language,)
    for language in languages:
        prepare_language(language)


if __name__ == "__main__":
    main()
