"""Shared loader: the 13,730 unique offers of the benchmark in German and English, keyed by offer id."""
import glob, gzip, json
import pandas as pd

FIELDS = ("brand", "name", "desc", "price")


def load_offers(language):
    recs = {}
    for f in sorted(glob.glob(f"data/solute_{language}/*/*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as h:
            df = pd.read_json(h, lines=True)
        for side in ("left", "right"):
            cols = [f"{c}_{side}" for c in FIELDS]
            for _, r in df[[f"id_{side}"] + cols].drop_duplicates(f"id_{side}").iterrows():
                recs.setdefault(str(r[f"id_{side}"]), {c: ("" if pd.isna(r[f"{c}_{side}"]) else str(r[f"{c}_{side}"])) for c in FIELDS})
    return recs


def aligned_offers():
    de, en = load_offers("de"), load_offers("en")
    assert set(de) == set(en), (len(de), len(en))
    return {i: {"de": de[i], "en": en[i]} for i in sorted(de)}


def write_jsonl(path, offers):
    with open(path, "w", encoding="utf-8") as h:
        for i, o in offers.items():
            h.write(json.dumps({"id": i, **o}, ensure_ascii=False) + "\n")
