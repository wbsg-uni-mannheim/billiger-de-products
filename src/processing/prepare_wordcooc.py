"""Create WordCooc features for all released benchmark variants."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


LANGUAGES = {
    "de": Path("data/processed"),
    "en": Path("data/processed_en"),
}
FEATURES = ("brand", "name", "price", "desc")
FEATURE_NAME = "+".join(FEATURES)


def combined_text(pairs, side):
    columns = [f"{attribute}_{side}" for attribute in FEATURES]
    return pairs[columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()


def add_features(pairs, vectorizer):
    pairs = pairs.copy()
    left = vectorizer.transform(combined_text(pairs, "left"))
    right = vectorizer.transform(combined_text(pairs, "right"))
    pairs[f"{FEATURE_NAME}_wordcooc"] = [
        left[index].multiply(right[index]).astype(int)
        for index in range(len(pairs))
    ]
    return pairs


def prepare_language(language):
    root = LANGUAGES[language]
    output_dir = root / "wordcooc/learning-curve"
    feature_dir = output_dir / "feature-names"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for train_path in sorted((root / "training-sets").glob("preprocessed_products*.pkl.gz")):
        valid_path = root / "validation-sets" / train_path.name.replace("train", "valid")
        train = pd.read_pickle(train_path)
        valid = pd.read_pickle(valid_path)
        development = pd.concat([train, valid], ignore_index=True)

        records = pd.concat(
            [
                development[["id_left"]].rename(columns={"id_left": "id"}).assign(text=combined_text(development, "left")),
                development[["id_right"]].rename(columns={"id_right": "id"}).assign(text=combined_text(development, "right")),
            ],
            ignore_index=True,
        ).drop_duplicates(subset="id")
        vectorizer = CountVectorizer(min_df=2, binary=True).fit(records["text"])

        train_name = train_path.name.removesuffix(".pkl.gz") + "_wordcooc.pkl.gz"
        valid_name = train_name.replace("train", "valid")
        development_features = add_features(development, vectorizer)
        development_features.to_pickle(output_dir / train_name, compression="gzip")
        development_features[development_features["pair_id"].isin(valid["pair_id"])].to_pickle(
            output_dir / valid_name,
            compression="gzip",
        )

        with open(feature_dir / train_name.replace(".pkl.gz", "_words.json"), "w", encoding="utf-8") as file:
            json.dump({FEATURE_NAME: vectorizer.get_feature_names_out().tolist()}, file)

        variant = train_path.name.split("_")[1]
        for unseen in ("000un", "050un", "100un"):
            test_variant = variant.replace("000un", unseen)
            test_path = root / "gold-standards_adjusted" / f"preprocessed_{test_variant}_gs.pkl.gz"
            test_name = train_name.replace(".pkl.gz", f"_preprocessed_{test_variant}_gs.pkl.gz")
            add_features(pd.read_pickle(test_path), vectorizer).to_pickle(
                output_dir / test_name,
                compression="gzip",
            )

    print(f"Prepared {language.upper()} WordCooc features in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("de", "en", "both"), default="both")
    args = parser.parse_args()

    languages = LANGUAGES if args.language == "both" else (args.language,)
    for language in languages:
        prepare_language(language)


if __name__ == "__main__":
    main()
