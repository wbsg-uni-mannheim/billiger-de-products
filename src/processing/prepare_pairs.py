"""Convert the released JSONL benchmark pairs to normalized pickle files."""

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

import utils


LANGUAGES = {
    "de": (Path("data/solute_de"), Path("data/processed")),
    "en": (Path("data/solute_en"), Path("data/processed_en")),
}
SPLITS = ("training-sets", "validation-sets", "gold-standards_adjusted")
TEXT_LIMITS = {"name": 50, "brand": 5, "desc": 100}


def normalize_text(value, limit):
    if pd.isna(value):
        return np.nan
    cleaned = utils.clean_string_2020(str(value))
    if cleaned is None:
        return np.nan
    cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split()[:limit])


def prepare_language(language):
    source_root, output_root = LANGUAGES[language]
    written = 0

    for split in SPLITS:
        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        for source in sorted((source_root / split).glob("products*.json.gz")):
            pairs = pd.read_json(source, lines=True, compression="gzip")
            for attribute, limit in TEXT_LIMITS.items():
                for side in ("left", "right"):
                    column = f"{attribute}_{side}"
                    if column in pairs:
                        pairs[column] = pairs[column].map(
                            lambda value, max_words=limit: normalize_text(value, max_words)
                        )

            output_name = f"preprocessed_{source.name.removesuffix('.json.gz')}.pkl.gz"
            pairs.reset_index(drop=True).to_pickle(
                output_dir / output_name,
                compression="gzip",
            )
            written += 1

    print(f"Prepared {written} {language.upper()} pair files in {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "both" else (args.language,)
    for language in languages:
        prepare_language(language)


if __name__ == "__main__":
    main()
