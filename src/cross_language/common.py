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
VALIDATION_VARIANT = "products80cc20rnd050un"
TRAIN_SIZE = "large"

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


def serialized_path(root, variant):
    return root / f"preprocessed_{BENCHMARK}_{variant}.txt"


def variant_from_path(path):
    stem = Path(path).name
    return next(variant for variant in VARIANTS if f"_{variant}." in stem)
