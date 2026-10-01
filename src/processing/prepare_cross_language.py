"""Create aligned cross-language test sets for the 80cc20rnd050un setup."""

import argparse
import html
import json
import random
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.cross_language.common import (
    BENCHMARK,
    DITTO_DIR,
    HIERGAT_DIR,
    PAIR_DIR,
    RAW_DIR,
    RSUPCON_PRETRAIN_DIR,
    SELECTION_SEEN_SHARE,
    SELECTION_VARIANT,
    TRAIN_SIZE,
    TRAIN_VARIANT,
    VALIDATION_VARIANT,
    VARIANTS,
    WORDCOOC_DIR,
    pair_path,
    raw_path,
    selection_pair_path,
    selection_serialized_path,
    serialized_path,
    validation_path,
)
from src.processing.prepare_wordcooc import add_features, combined_text
from src.processing.prepare_pretraining import extract_records


LANGUAGE_COMBINATIONS = {
    "de_de": ("de", "de"),
    "de_en": ("de", "en"),
    "en_de": ("en", "de"),
    "en_en": ("en", "en"),
}
TEXT_LIMITS = {"name": 50, "brand": 5, "desc": 100}
ATTRIBUTES = ("brand", "name", "price", "desc")


def normalize_text(value, limit):
    if pd.isna(value):
        return np.nan
    cleaned = html.unescape(re.sub(r"<[^>]+>", " ", str(value)))
    return " ".join(cleaned.split()[:limit])


def read_aligned_pairs():
    german = pd.read_json(
        f"data/solute_de/gold-standards_adjusted/{BENCHMARK}.json.gz",
        lines=True,
        compression="gzip",
    )
    english = pd.read_json(
        f"data/solute_en/gold-standards_adjusted/{BENCHMARK}.json.gz",
        lines=True,
        compression="gzip",
    )

    if not german["pair_id"].equals(english["pair_id"]):
        raise ValueError("German and English pair order or pair IDs differ")

    fixed_columns = (
        "pair_id",
        "label",
        "is_hard_negative",
        "id_left",
        "id_right",
        "product_id_left",
        "product_id_right",
    )
    for column in fixed_columns:
        if not german[column].equals(english[column]):
            raise ValueError(f"German and English values differ in {column}")

    return {"de": german, "en": english}


def random_languages(pairs, seed):
    assignments = pd.DataFrame(index=pairs.index, columns=("left", "right"))
    combinations = list(LANGUAGE_COMBINATIONS.values())
    combination_counts = dict.fromkeys(combinations, 0)

    grouped = pairs.groupby(["label", "is_hard_negative"], sort=True, dropna=False)
    for group_number, (_, group) in enumerate(grouped):
        rng = random.Random(seed + group_number)
        indices = list(group.index)
        rng.shuffle(indices)

        base_count, remainder = divmod(len(indices), len(combinations))
        rotated = (
            combinations[group_number % len(combinations) :]
            + combinations[: group_number % len(combinations)]
        )
        remainder_order = sorted(
            rotated,
            key=lambda combination: combination_counts[combination],
        )
        values = combinations * base_count + remainder_order[:remainder]
        for combination in values:
            combination_counts[combination] += 1

        rng.shuffle(values)
        for index, (left, right) in zip(indices, values):
            assignments.loc[index] = (left, right)

    return assignments


def combine_pairs(aligned, left_languages, right_languages, variant):
    combined = aligned["de"].copy()
    side_columns = {
        side: [column for column in combined.columns if column.endswith(f"_{side}")]
        for side in ("left", "right")
    }

    for index in combined.index:
        for side, language in (
            ("left", left_languages.loc[index]),
            ("right", right_languages.loc[index]),
        ):
            combined.loc[index, side_columns[side]] = aligned[language].loc[
                index, side_columns[side]
            ]

    combined["language_left"] = left_languages
    combined["language_right"] = right_languages
    combined["test_variant"] = variant
    return combined


def normalize_pairs(pairs):
    normalized = pairs.copy()
    for attribute, limit in TEXT_LIMITS.items():
        for side in ("left", "right"):
            column = f"{attribute}_{side}"
            normalized[column] = normalized[column].map(
                lambda value, max_words=limit: normalize_text(value, max_words)
            )
    return normalized


def serialize_record(row, side):
    return " ".join(
        f"COL {attribute} VAL {'' if pd.isna(row[f'{attribute}_{side}']) else row[f'{attribute}_{side}']}"
        for attribute in ATTRIBUTES
    )


def write_serialized_pairs(pairs, output):
    left = pairs.apply(serialize_record, axis=1, side="left")
    right = pairs.apply(serialize_record, axis=1, side="right")
    labels = pairs["label"].astype(int).astype(str)
    serialized = left + "\t" + right + "\t" + labels
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(serialized.astype(str)) + "\n", encoding="utf-8")


def unordered_offer_pair_keys(pairs):
    return [
        tuple(sorted((str(left), str(right))))
        for left, right in zip(pairs["id_left"], pairs["id_right"])
    ]


def prepare_selection_pairs():
    """Load the split the cross-language jobs select checkpoints on.

    This is the very same file the main benchmark grid validates on, so no copy
    is written: the run scripts point --validation_file straight at
    data/processed/.  The checks below only assert that the file has the
    properties the protocol claims for it.
    """
    train = pd.read_pickle(
        "data/processed/training-sets/"
        f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}.pkl.gz"
    )
    selection = pd.read_pickle(selection_pair_path())

    train_pair_keys = set(unordered_offer_pair_keys(train))
    overlap = train_pair_keys & set(unordered_offer_pair_keys(selection))
    if overlap:
        raise ValueError(
            f"{selection_pair_path()} shares {len(overlap)} offer pairs with training"
        )

    train_products = set(train["product_id_left"]) | set(train["product_id_right"])
    selection_products = (
        set(selection["product_id_left"]) | set(selection["product_id_right"])
    )
    seen_share = len(train_products & selection_products) / len(selection_products)
    if seen_share != SELECTION_SEEN_SHARE:
        raise ValueError(
            f"Selection seen-product share is {seen_share}, "
            f"expected {SELECTION_SEEN_SHARE}"
        )

    for model in ("ditto", "hiergat"):
        serialized = selection_serialized_path(model)
        if not serialized.is_file():
            raise ValueError(
                f"{serialized} is missing; run prepare_ditto_hiergat.py first"
            )
        rows = sum(
            1
            for line in serialized.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if rows != len(selection):
            raise ValueError(
                f"{serialized} has {rows} rows, expected {len(selection)}"
            )

    return train, selection


def prepare_development_pairs():
    train = pd.read_pickle(
        "data/processed/training-sets/"
        f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}.pkl.gz"
    )
    valid = pd.read_pickle(
        "data/processed/validation-sets/"
        f"preprocessed_{VALIDATION_VARIANT}_valid_{TRAIN_SIZE}.pkl.gz"
    )

    train_pair_keys = set(unordered_offer_pair_keys(train))
    keep_validation = [
        pair_key not in train_pair_keys
        for pair_key in unordered_offer_pair_keys(valid)
    ]
    valid = valid.loc[keep_validation].reset_index(drop=True)

    remaining_overlap = train_pair_keys & set(unordered_offer_pair_keys(valid))
    if remaining_overlap:
        raise ValueError("Training and validation still contain identical offer pairs")

    train_products = set(train["product_id_left"]) | set(train["product_id_right"])
    valid_products = set(valid["product_id_left"]) | set(valid["product_id_right"])
    seen_share = len(train_products & valid_products) / len(valid_products)
    if seen_share != 0.5:
        raise ValueError(f"Validation seen-product share is {seen_share}, expected 0.5")

    validation_path().parent.mkdir(parents=True, exist_ok=True)
    valid.to_pickle(validation_path(), compression="gzip")
    validation_name = (
        f"preprocessed_{VALIDATION_VARIANT}_valid_{TRAIN_SIZE}.txt"
    )
    write_serialized_pairs(valid, DITTO_DIR / validation_name)
    write_serialized_pairs(valid, HIERGAT_DIR / validation_name)
    return train, valid


def prepare_wordcooc(train, selection, cross_pairs):
    """Fit the WordCooc vocabulary on exactly the German development set the
    main benchmark run uses (training split + SELECTION_VARIANT validation
    split), then project the five cross-language test sets onto it."""
    development = pd.concat([train, selection], ignore_index=True)
    records = pd.concat(
        [
            development[["id_left"]]
            .rename(columns={"id_left": "id"})
            .assign(text=combined_text(development, "left")),
            development[["id_right"]]
            .rename(columns={"id_right": "id"})
            .assign(text=combined_text(development, "right")),
        ],
        ignore_index=True,
    ).drop_duplicates(subset="id")
    vectorizer = CountVectorizer(min_df=2, binary=True).fit(records["text"])

    WORDCOOC_DIR.mkdir(parents=True, exist_ok=True)
    feature_dir = WORDCOOC_DIR / "feature-names"
    feature_dir.mkdir(parents=True, exist_ok=True)
    train_name = (
        f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}_wordcooc.pkl.gz"
    )
    valid_name = (
        f"preprocessed_{SELECTION_VARIANT}_valid_{TRAIN_SIZE}_wordcooc.pkl.gz"
    )
    development_features = add_features(development, vectorizer)
    development_features.to_pickle(
        WORDCOOC_DIR / train_name,
        compression="gzip",
    )
    development_features[
        development_features["pair_id"].isin(selection["pair_id"])
    ].to_pickle(
        WORDCOOC_DIR / valid_name,
        compression="gzip",
    )
    (feature_dir / train_name.replace(".pkl.gz", "_words.json")).write_text(
        json.dumps(
            {
                "brand+name+price+desc": vectorizer.get_feature_names_out().tolist()
            }
        ),
        encoding="utf-8",
    )

    for variant, pairs in cross_pairs.items():
        add_features(pairs, vectorizer).to_pickle(
            WORDCOOC_DIR / f"preprocessed_{BENCHMARK}_{variant}_wordcooc.pkl.gz",
            compression="gzip",
        )


def prepare_rsupcon_pretraining(train):
    records = extract_records(train)
    RSUPCON_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    records.to_pickle(
        RSUPCON_PRETRAIN_DIR
        / f"{TRAIN_VARIANT}_train_{TRAIN_SIZE}.pkl.gz",
        compression="gzip",
    )


def validate_outputs(cross_pairs):
    reference = cross_pairs["de_de"]
    for variant, pairs in cross_pairs.items():
        for column in (
            "pair_id",
            "label",
            "is_hard_negative",
            "id_left",
            "id_right",
            "product_id_left",
            "product_id_right",
        ):
            if not pairs[column].equals(reference[column]):
                raise ValueError(f"{variant} differs from DE-DE in {column}")

    random_counts = (
        cross_pairs["random"]
        .groupby(["language_left", "language_right"])
        .size()
        .to_dict()
    )
    if max(random_counts.values()) - min(random_counts.values()) > 2:
        raise ValueError(f"Random-Random language combinations are unbalanced: {random_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    aligned = read_aligned_pairs()
    train, selection = prepare_selection_pairs()
    _, valid = prepare_development_pairs()
    random_assignment = random_languages(aligned["de"], args.seed)
    cross_pairs = {}

    for variant in VARIANTS:
        if variant == "random":
            left_languages = random_assignment["left"]
            right_languages = random_assignment["right"]
        else:
            left_language, right_language = LANGUAGE_COMBINATIONS[variant]
            left_languages = pd.Series(left_language, index=aligned["de"].index)
            right_languages = pd.Series(right_language, index=aligned["de"].index)

        raw_pairs = combine_pairs(
            aligned,
            left_languages,
            right_languages,
            variant,
        )
        normalized = normalize_pairs(raw_pairs)
        cross_pairs[variant] = normalized

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_pairs.to_json(
            raw_path(variant),
            orient="records",
            lines=True,
            compression="gzip",
            force_ascii=False,
        )

        PAIR_DIR.mkdir(parents=True, exist_ok=True)
        normalized.to_pickle(pair_path(variant), compression="gzip")
        write_serialized_pairs(normalized, serialized_path(DITTO_DIR, variant))
        write_serialized_pairs(normalized, serialized_path(HIERGAT_DIR, variant))

    validate_outputs(cross_pairs)
    prepare_wordcooc(train, selection, cross_pairs)
    prepare_rsupcon_pretraining(train)

    counts = (
        cross_pairs["random"]
        .groupby(["language_left", "language_right"])
        .size()
    )
    print(f"Prepared {len(cross_pairs)} cross-language test sets with {len(aligned['de'])} pairs each")
    print(
        f"Selection split {selection_pair_path()} "
        f"({SELECTION_VARIANT}): {len(selection)} pairs"
    )
    print(f"Prepared {len(valid)} disjoint DE-DE {VALIDATION_VARIANT} validation pairs")
    print(counts.to_string())


if __name__ == "__main__":
    main()
