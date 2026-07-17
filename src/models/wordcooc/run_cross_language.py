"""Run the German-trained WordCooc baseline on all cross-language test sets."""

from src.cross_language.common import (
    BENCHMARK,
    TRAIN_SIZE,
    TRAIN_VARIANT,
    VALIDATION_VARIANT,
    VARIANTS,
    WORDCOOC_DIR,
)
from src.models.wordcooc import run_wordcooc as benchmark


def main():
    train = (
        WORDCOOC_DIR
        / f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}_wordcooc.pkl.gz"
    )
    valid = (
        WORDCOOC_DIR
        / f"preprocessed_{VALIDATION_VARIANT}_valid_{TRAIN_SIZE}_wordcooc.pkl.gz"
    )
    benchmark.RESULT_ROOT = "results/generated/cross_language/wordcooc"

    for variant in VARIANTS:
        test = WORDCOOC_DIR / f"preprocessed_{BENCHMARK}_{variant}_wordcooc.pkl.gz"
        benchmark.run_wordcooc(
            str(train),
            str(valid),
            str(test),
            ["brand+name+price+desc"],
            benchmark.classifiers,
            "de_train_000un_valid_test_050un",
            write_test_set_for_inspection=False,
        )


if __name__ == "__main__":
    main()
