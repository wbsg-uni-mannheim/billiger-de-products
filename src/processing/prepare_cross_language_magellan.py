"""Create German development and cross-language test tables for Magellan."""

import pandas as pd

from src.cross_language.common import (
    MAGELLAN_DIR,
    TRAIN_SIZE,
    TRAIN_VARIANT,
    VALIDATION_VARIANT,
    VARIANTS,
    pair_path,
    validation_path,
)
from src.processing.prepare_magellan import write_tables


def main():
    MAGELLAN_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_pickle(
        "data/processed/training-sets/"
        f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}.pkl.gz"
    )
    valid = pd.read_pickle(validation_path())
    development = pd.concat([train, valid], ignore_index=True)
    development_stem = (
        f"preprocessed_{VALIDATION_VARIANT}_train_{TRAIN_SIZE}_cross_magellan_"
    )
    write_tables(
        development,
        MAGELLAN_DIR,
        development_stem,
        set(valid["pair_id"]),
    )

    for variant in VARIANTS:
        stem = f"preprocessed_products80cc20rnd050un_gs_{variant}_magellan_"
        write_tables(pd.read_pickle(pair_path(variant)), MAGELLAN_DIR, stem)
    print(
        f"Prepared DE-DE {VALIDATION_VARIANT} validation and "
        f"{len(VARIANTS)} cross-language Magellan test sets"
    )


if __name__ == "__main__":
    main()
