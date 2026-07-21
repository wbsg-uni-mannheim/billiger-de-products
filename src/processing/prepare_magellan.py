"""Create py_entitymatching tables for all benchmark variants."""

import argparse
from pathlib import Path

import pandas as pd
import py_entitymatching as em


LANGUAGES = {
    "de": Path("data/processed"),
    "en": Path("data/processed_en"),
}


def rename_pair_columns(pairs):
    return pairs.rename(
        columns={
            column: f"ltable_{column.removesuffix('_left')}"
            if column.endswith("_left")
            else f"rtable_{column.removesuffix('_right')}"
            if column.endswith("_right")
            else column
            for column in pairs.columns
        }
    )


def write_tables(pairs, output_dir, stem, validation_ids=None):
    pairs = rename_pair_columns(pairs.fillna(""))
    left = pairs.filter(regex="^ltable_").drop_duplicates(subset="ltable_id").copy()
    right = pairs.filter(regex="^rtable_").drop_duplicates(subset="rtable_id").copy()
    left["mag_id"] = range(len(left))
    right["mag_id"] = range(len(right))

    pairs = pairs.merge(left[["ltable_id", "mag_id"]], on="ltable_id").rename(columns={"mag_id": "ltable_mag_id"})
    pairs = pairs.merge(right[["rtable_id", "mag_id"]], on="rtable_id").rename(columns={"mag_id": "rtable_mag_id"})
    pairs["_id"] = range(len(pairs))

    left = left.drop(columns="ltable_id").rename(columns=lambda column: column.removeprefix("ltable_"))
    right = right.drop(columns="rtable_id").rename(columns=lambda column: column.removeprefix("rtable_"))

    left_path = output_dir / f"{stem}left_formatted.csv"
    right_path = output_dir / f"{stem}right_formatted.csv"
    pair_path = output_dir / f"{stem}pairs_formatted.csv"
    left.to_csv(left_path, index=False)
    right.to_csv(right_path, index=False)
    left_meta = em.read_csv_metadata(str(left_path), key="mag_id")
    right_meta = em.read_csv_metadata(str(right_path), key="mag_id")
    em.to_csv_metadata(left_meta, str(left_path))
    em.to_csv_metadata(right_meta, str(right_path))

    pairs.to_csv(pair_path, index=False)
    pair_meta = em.read_csv_metadata(
        str(pair_path),
        key="_id",
        ltable=left_meta,
        rtable=right_meta,
        fk_ltable="ltable_mag_id",
        fk_rtable="rtable_mag_id",
    )
    em.to_csv_metadata(pair_meta, str(pair_path))

    if validation_ids is not None:
        is_validation = pair_meta["pair_id"].isin(validation_ids)
        em.to_csv_metadata(pair_meta[~is_validation], str(output_dir / pair_path.name.replace("train", "trainonly")))
        em.to_csv_metadata(pair_meta[is_validation], str(output_dir / pair_path.name.replace("train", "valid")))


def prepare_language(language):
    root = LANGUAGES[language]
    output_dir = root / "magellan/learning-curve/formatted"
    output_dir.mkdir(parents=True, exist_ok=True)

    for train_path in sorted((root / "training-sets").glob("preprocessed_products*.pkl.gz")):
        valid_path = root / "validation-sets" / train_path.name.replace("train", "valid")
        train = pd.read_pickle(train_path)
        valid = pd.read_pickle(valid_path)
        development = pd.concat([train, valid], ignore_index=True)
        stem = train_path.name.removesuffix(".pkl.gz") + "_magellan_"
        write_tables(development, output_dir, stem, set(valid["pair_id"]))

    for test_path in sorted((root / "gold-standards_adjusted").glob("preprocessed_products*.pkl.gz")):
        stem = test_path.name.removesuffix(".pkl.gz") + "_magellan_"
        write_tables(pd.read_pickle(test_path), output_dir, stem)

    print(f"Prepared {language.upper()} Magellan tables in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "both" else (args.language,)
    for language in languages:
        prepare_language(language)


if __name__ == "__main__":
    main()
