from pathlib import Path


VARIANTS = ("de_de", "de_en", "en_de", "en_en", "random")
DISPLAY_NAMES = {
    "de_de": "DE-DE",
    "de_en": "DE-EN",
    "en_de": "EN-DE",
    "en_en": "EN-EN",
    "random": "Random-Random",
}

BENCHMARK = "products80cc20rnd050un_gs"
TRAIN_VARIANT = "products80cc20rnd000un"

# Split used to pick the reported checkpoint / hyper-parameters.  It has to be
# the split the main benchmark grid selects on, otherwise the DE-DE column of
# the cross-language table and the 80cc20 / Large / Half-Seen cell of the main
# table are two different experiments.  run_confidence_test.sh,
# run_finetune_siamese.sh, all_runs_de.py and hier_de.sh all select on the
# TRAIN_VARIANT validation split, so the cross-language jobs do the same and
# read the very same files out of data/processed/.
SELECTION_VARIANT = TRAIN_VARIANT
SELECTION_SEEN_SHARE = 1.0

# Half-seen validation split.  The released benchmark documentation describes
# this as the selection split for the 050un and 100un conditions, but no
# benchmark run actually uses it.  It is still prepared and asserted here so
# that the alternative protocol stays one flag away; see
# reports/table3_table6_consistency.md.
VALIDATION_VARIANT = "products80cc20rnd050un"
VALIDATION_SEEN_SHARE = 0.5

TRAIN_SIZE = "large"

# Directory name the classical matchers write their result tables into.  It
# spells out all three splits so that a result file cannot be mistaken for one
# produced under the half-seen selection protocol.
EXPERIMENT_NAME = "de_train_000un_valid_000un_test_050un"

ROOT = Path("data/processed_cross_language")
RAW_DIR = ROOT / "raw"
PAIR_DIR = ROOT / "gold-standards_adjusted"
VALIDATION_DIR = ROOT / "validation-sets"
DITTO_DIR = ROOT / "ditto/data/final_output"
HIERGAT_DIR = ROOT / "hiergat/data/final_output"
WORDCOOC_DIR = ROOT / "wordcooc"
MAGELLAN_DIR = ROOT / "magellan"
RSUPCON_PRETRAIN_DIR = ROOT / "r-supcon/pretrain"


def raw_path(variant):
    return RAW_DIR / f"{BENCHMARK}_{variant}.json.gz"


def pair_path(variant):
    return PAIR_DIR / f"preprocessed_{BENCHMARK}_{variant}.pkl.gz"


def validation_path():
    return (
        VALIDATION_DIR
        / f"preprocessed_{VALIDATION_VARIANT}_valid_{TRAIN_SIZE}.pkl.gz"
    )


def train_pair_path():
    return Path(
        "data/processed/training-sets/"
        f"preprocessed_{TRAIN_VARIANT}_train_{TRAIN_SIZE}.pkl.gz"
    )


def selection_pair_path():
    """Validation split used for model selection, read from the main pipeline."""
    return Path(
        "data/processed/validation-sets/"
        f"preprocessed_{SELECTION_VARIANT}_valid_{TRAIN_SIZE}.pkl.gz"
    )


def selection_serialized_path(model):
    """Ditto / HierGAT selection split, read from the main pipeline."""
    return Path(
        f"data/processed/{model}/data/final_output/"
        f"preprocessed_{SELECTION_VARIANT}_valid_{TRAIN_SIZE}.txt"
    )


def serialized_path(root, variant):
    return root / f"preprocessed_{BENCHMARK}_{variant}.txt"


def variant_from_path(path):
    stem = Path(path).name
    return next(variant for variant in VARIANTS if f"_{variant}." in stem)
