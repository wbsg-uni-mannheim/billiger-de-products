"""Run the German-trained Magellan baseline on all cross-language test sets."""

from src.cross_language.common import (
    EXPERIMENT_NAME,
    MAGELLAN_DIR,
    SELECTION_VARIANT,
    TRAIN_SIZE,
    VARIANTS,
)
from src.models.magellan import run_magellan as benchmark


def main():
    train = (
        MAGELLAN_DIR
        / f"preprocessed_{SELECTION_VARIANT}_train_{TRAIN_SIZE}"
        "_cross_magellan_pairs_formatted.csv"
    )
    valid = (
        MAGELLAN_DIR
        / f"preprocessed_{SELECTION_VARIANT}_valid_{TRAIN_SIZE}"
        "_cross_magellan_pairs_formatted.csv"
    )
    benchmark.RESULT_ROOT = "results/generated/cross_language/magellan"

    for variant in VARIANTS:
        test = (
            MAGELLAN_DIR
            / f"preprocessed_products80cc20rnd050un_gs_{variant}_magellan_pairs_formatted.csv"
        )
        benchmark.run_magellan(
            str(train),
            str(valid),
            str(test),
            [["brand", "name", "desc", "price"]],
            benchmark.classifiers,
            EXPERIMENT_NAME,
            write_test_set_for_inspection=False,
        )


if __name__ == "__main__":
    main()
