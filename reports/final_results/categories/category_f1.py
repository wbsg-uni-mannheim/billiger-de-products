"""Same-category F1 per model on the German v1.1 test sets (category table in the paper).

Inputs
- reports/final_results/categories/offer_categories.csv: offer_id -> top-level category for the 13,568 benchmark
  offers, produced by construction/categories/category_extraction.py.
- <export json> (argv[2]): per-run export written by src/v11/export_comparison.py (scored file and, for
  WordCooc/Magellan, the selected classifier).
- <prediction dir> (argv[1]): per-pair prediction CSVs (pair_id,label,prediction) of the v1.1 runs.
- reports/final_results/collapsed_runs.csv: runs excluded by the collapse rule (see reports/final_results/README.md).

Aggregation: same-category pairs only, F1 per (model, test set, size, seed), mean over (size, seed) within each of
the nine German test sets, then mean over the test sets in which the category occurs.

Run from the repository root:

    python reports/final_results/categories/category_f1.py <prediction dir> <export json>

Outputs (next to this script): category_f1.csv (table), category_f1_long.csv (per run), category_counts.csv.
"""

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EV = Path(__file__).resolve().parent
PRED = Path(sys.argv[1])
SERVER_PREFIX = re.compile(r"^.*?/results/generated/")  # export paths are absolute on the cluster
MODELS = ["wordcooc", "magellan", "roberta", "r-supcon", "hiergat", "ditto", "gpt-5.2-simple", "gpt-5.2-rule-guided"]
TESTS = {"Seen": "000", "Half-Seen": "050", "Unseen": "100"}
CAT_EN = {"Kleidung & Accessoires": "Clothing & Accessories", "Möbel & Wohnen": "Furniture & Living",
          "Elektronik & Computer": "Electronics & Computers", "Werkzeug & Baumarkt": "Tools & DIY",
          "Sport & Freizeit": "Sports & Leisure", "Spielzeug & Baby": "Toys & Baby",
          "Auto & Motorrad": "Automotive & Motorcycle", "Kosmetik & Drogerie": "Cosmetics & Drugstore",
          "Gesundheit & Pflege": "Health & Care", "Bürobedarf": "Office Supplies",
          "Lebensmittel & Getränke": "Food & Beverages", "Bücher, Filme & Musik": "Books, Films & Music",
          "Haustier & Tierbedarf": "Pet Supplies"}

cats = pd.read_csv(EV / "offer_categories.csv").set_index("offer_id").category_de
export = json.loads(Path(sys.argv[2]).read_text())
rows = [r for r in export["rows"] if r["table"] == "main_grid" and r["language"] == "de" and r["model"] in MODELS]
assert len(rows) == 6 * 81 + 2 * 9, len(rows)


def local_path(r):
    src = SERVER_PREFIX.sub("", r["source"]).removeprefix("results/generated/")
    m = r["model"]
    if m in ("roberta", "r-supcon"):
        return PRED / src.replace("_metrics.json", "_predictions.csv")
    if m in ("ditto", "hiergat"):
        return PRED / src
    if m.startswith("gpt"):
        return PRED / Path(src).parent / "predictions.csv"
    folder = PRED / Path(src).parent.parent / "predictions"
    gs = re.search(r"(preprocessed_products\d+cc\d+rnd\d+un_gs)", Path(src).name).group(1)
    if m == "wordcooc":
        train = re.search(r"(preprocessed_products\d+cc\d+rnd000un_train_\w+?)_wordcooc", Path(src).name).group(1)
        return folder / f"{train}_wordcooc_{gs}_{r['classifier']}_brand+name+price+desc_run{r['seed']}.csv"
    return folder / f"{gs}_magellan_pairs_formatted_{r['classifier']}_run{r['seed']}.csv"


def f1(df):
    tp = ((df.label == 1) & (df.prediction == 1)).sum()
    fp = ((df.label == 0) & (df.prediction == 1)).sum()
    fn = ((df.label == 1) & (df.prediction == 0)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


long, checks = [], []
for r in rows:
    p = local_path(r)
    df = pd.read_csv(p)[["pair_id", "label", "prediction"]]
    df["prediction"] = df.prediction.fillna(0).astype(int)
    ids = df.pair_id.str.split("#", expand=True).astype(int)
    df["cat_left"], df["cat_right"] = cats.reindex(ids[0]).values, cats.reindex(ids[1]).values
    assert df.cat_left.notna().all() and df.cat_right.notna().all(), p
    total = 100 * f1(df)
    checks.append(dict(model=r["model"], cc=r["cc"], size=r["size"], test=r["test"], seed=r["seed"], export_f1=r["f1"],
                       file_f1=total, pairs=len(df), ok=abs(total - r["f1"]) < 0.01))
    same = df[df.cat_left == df.cat_right]
    for cat, g in same.groupby("cat_left"):
        long.append(dict(model=r["model"], cc=int(r["cc"]), un=TESTS[r["test"]], size=r["size"] or "", seed=r["seed"],
                         category_de=cat, category=CAT_EN[cat], f1=f1(g), pairs=len(g), positives=int(g.label.sum())))

checks = pd.DataFrame(checks)
assert checks.ok.all(), checks[~checks.ok]
long = pd.DataFrame(long)
collapsed = pd.read_csv(ROOT / "reports/final_results/collapsed_runs.csv")
collapsed = collapsed[(collapsed.table == "main_grid") & (collapsed.language == "de") & (collapsed.action == "excluded")]
excluded = set(zip(collapsed.model, collapsed.cc, collapsed["size"], collapsed.seed))
long["excluded_collapsed"] = [k in excluded for k in zip(long.model, long.cc, long["size"], long.seed)]
long.to_csv(EV / "category_f1_long.csv", index=False)
kept = long[~long.excluded_collapsed]
table = (kept.groupby(["category", "model", "cc", "un"]).f1.mean().groupby(["category", "model"]).mean().unstack("model")[MODELS] * 100)
n_tests = long.groupby("category")[["cc", "un"]].apply(lambda g: g.drop_duplicates().shape[0])
table["test_sets_present"] = n_tests
table.round(2).to_csv(EV / "category_f1.csv")

# Record occurrences (2 per pair) per category in the 80% large training set and the v1.1 80% Half-Seen test set.
def occurrences(path):
    c = {}
    for line in gzip.open(path, "rt"):
        rec = json.loads(line)
        for k in ("id_left", "id_right"):
            cat = CAT_EN[cats[rec[k]]]
            c[cat] = c.get(cat, 0) + 1
    return pd.Series(c)


data = ROOT / "data/solute_de"
counts = pd.DataFrame({
    "train_large_80cc": occurrences(data / "training-sets/products80cc20rnd000un_train_large.json.gz"),
    "test_half_seen_80cc": occurrences(data / "gold-standards_adjusted/products80cc20rnd050un_gs.json.gz"),
}).fillna(0).astype(int).sort_values("train_large_80cc", ascending=False)
counts.to_csv(EV / "category_counts.csv")

print(f"all {len(checks)} prediction files reproduce the exported cell F1")
print(table.round(2).to_string())
print(counts.to_string())
