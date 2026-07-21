"""Serialize normalized pairs for Ditto and HierGAT."""

import argparse
from pathlib import Path

import pandas as pd


LANGUAGES = {
    "de": Path("data/processed"),
    "en": Path("data/processed_en"),
}
SPLITS = ("training-sets", "validation-sets", "gold-standards_adjusted")
ATTRIBUTES = ("brand", "name", "price", "desc")


def serialize_record(row, side):
    return " ".join(
        f"COL {attribute} VAL {'' if pd.isna(row[f'{attribute}_{side}']) else row[f'{attribute}_{side}']}"
        for attribute in ATTRIBUTES
    )


def serialize_pairs(path):
    pairs = pd.read_pickle(path)
    left = pairs.apply(serialize_record, axis=1, side="left")
    right = pairs.apply(serialize_record, axis=1, side="right")
    labels = pairs["label"].astype(int).astype(str)
    return left + "\t" + right + "\t" + labels


def prepare_language(language):
    root = LANGUAGES[language]
    outputs = [
        root / "ditto/data/final_output",
        root / "hiergat/data/final_output",
    ]
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)

    written = 0
    for split in SPLITS:
        for source in sorted((root / split).glob("preprocessed_products*.pkl.gz")):
            serialized = serialize_pairs(source)
            output_name = f"{source.name.removesuffix('.pkl.gz')}.txt"
            for output in outputs:
                (output / output_name).write_text(
                    "\n".join(serialized.astype(str)) + "\n", encoding="utf-8"
                )
            written += 1

    print(f"Prepared {written} {language.upper()} Ditto/HierGAT files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "both" else (args.language,)
    for language in languages:
        prepare_language(language)


if __name__ == "__main__":
    main()
