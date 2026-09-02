"""Summarize the XLM-R baseline runs into paper-ready rows.

Two inputs, both produced by run_finetune_baseline.py:

* main grid: results/generated/xlmr/{de,en}/products80cc20rnd000un-{size}/{seed}/
  with baseline_predictions{,_un050,_un100}.csv (pair_id, label, probability,
  prediction). These are re-scored against the RELEASED gold standards
  (data/solute_{de,en}/gold-standards_adjusted) by joining on pair_id, so the
  numbers are on the same basis as the rescored Ditto and HierGAT cells.
* cross-language: results/generated/cross_language/xlmr/80cc20-large/{seed}/
  predict_cross_{variant}_results.json (the aligned test pickles already carry
  the released labels).

Outputs under results/generated/xlmr/: main_per_seed.csv, main_mean.csv,
cross_per_seed.csv, cross_mean.csv, and the LaTeX rows on stdout. Pass
--compare_root to print the same rows for another backbone laid out the same
way (e.g. results/generated/roberta_bs32_full).

    python src/models/transformer_bert_confidence/summarize_xlmr.py
"""

import argparse
import gzip
import json
import statistics
from pathlib import Path

import pandas as pd

CATEGORY = "products80cc20rnd000un"
SIZES = ("small", "medium", "large")
TESTS = (("000un", "baseline_predictions.csv", "Seen"),
         ("050un", "baseline_predictions_un050.csv", "Half-Seen"),
         ("100un", "baseline_predictions_un100.csv", "Unseen"))
VARIANTS = ("de_de", "de_en", "en_de", "en_en", "random")
GOLD = {"de": Path("data/solute_de/gold-standards_adjusted"),
        "en": Path("data/solute_en/gold-standards_adjusted")}


def released_labels(language, un):
    path = GOLD[language] / f"products80cc20rnd{un}_gs.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        gold = pd.read_json(handle, lines=True)
    gold["pair_id"] = gold["pair_id"].astype(str)
    return gold[["pair_id", "label"]]


def prf(y, yhat):
    """Precision, recall and F1 on the match class, zero when undefined."""
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pred_csv, gold):
    pred = pd.read_csv(pred_csv, dtype={"pair_id": str})
    joined = gold.merge(pred[["pair_id", "prediction"]], on="pair_id", how="left", indicator=True)
    missing = int((joined["_merge"] == "left_only").sum())
    assert missing == 0, f"{missing} released pairs without a prediction in {pred_csv}"
    y, yhat = joined["label"].astype(int).values, joined["prediction"].astype(int).values
    p, r, f = prf(y, yhat)
    return 100 * p, 100 * r, 100 * f, len(joined), len(pred) - len(joined)


def main_grid(root):
    rows = []
    for language in ("de", "en"):
        gold = {un: released_labels(language, un) for un, _, _ in TESTS}
        for size in SIZES:
            for seed_dir in sorted((root / language / f"{CATEGORY}-{size}").glob("[0-9]*")):
                if not seed_dir.is_dir():
                    continue
                for un, name, test in TESTS:
                    csv = seed_dir / name
                    if not csv.exists():
                        print(f"missing {csv}")
                        continue
                    p, r, f, n, extra = score(csv, gold[un])
                    rows.append({"model": root.name, "language": language, "size": size, "test": test,
                                 "seed": int(seed_dir.name), "precision": p, "recall": r, "f1": f,
                                 "n_pairs": n, "n_dropped_predictions": extra})
    return pd.DataFrame(rows)


def cross_language(root):
    rows = []
    for seed_dir in sorted(root.glob("80cc20-large/[0-9]*")):
        for variant in VARIANTS:
            path = seed_dir / f"predict_cross_{variant}_results.json"
            if not path.exists():
                print(f"missing {path}")
                continue
            m = json.loads(path.read_text())
            key = f"predict_cross_{variant}"
            rows.append({"model": root.name, "variant": variant, "seed": int(seed_dir.name),
                         "precision": 100 * m[f"{key}_precision"], "recall": 100 * m[f"{key}_recall"],
                         "f1": 100 * m[f"{key}_f1"], "n_pairs": m.get(f"{key}_samples", "")})
    return pd.DataFrame(rows)


def aggregate(df, keys):
    if df.empty:
        return df
    g = df.groupby(keys)
    out = g["f1"].agg(f1_mean="mean", f1_std=lambda s: statistics.pstdev(s) if len(s) > 1 else 0.0,
                      n_seeds="count").reset_index()
    out["precision_mean"] = g["precision"].mean().values
    out["recall_mean"] = g["recall"].mean().values
    out["seeds"] = g["seed"].apply(lambda s: ",".join(str(x) for x in sorted(s))).values
    return out


def latex_main(mean, label):
    print(f"% {label}: F1 (mean over seeds, std in brackets) on the 80 % corner-case variants")
    print("% Size & Test & DE & EN")
    for size in SIZES:
        for _, _, test in TESTS:
            cells = []
            for language in ("de", "en"):
                r = mean[(mean.language == language) & (mean["size"] == size) & (mean.test == test)]
                cells.append(f"{r.f1_mean.iloc[0]:.2f} ({r.f1_std.iloc[0]:.2f})" if len(r) else "--")
            print(f"{size.capitalize()} & {test} & " + " & ".join(cells) + r" \\")


def latex_cross(mean, label):
    if mean.empty:
        return
    base = mean[mean.variant == "de_de"].f1_mean.iloc[0]
    cells = [f"{base:.1f}"] + [f"${mean[mean.variant == v].f1_mean.iloc[0] - base:+.1f}$"
                               for v in ("de_en", "en_de", "en_en", "random")]
    print(f"% {label}: DE-DE F1 and difference to DE-DE for DE-EN, EN-DE, EN-EN, Mixed")
    print(f"{label} & " + " & ".join(cells) + r" \\")


def run(main_root, cross_root, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = main_grid(main_root)
    mean = aggregate(per_seed, ["model", "language", "size", "test"])
    per_seed.to_csv(out_dir / f"{label}_main_per_seed.csv", index=False)
    mean.to_csv(out_dir / f"{label}_main_mean.csv", index=False)
    if not mean.empty:
        latex_main(mean, label)
    cross = cross_language(cross_root)
    cross_mean = aggregate(cross, ["model", "variant"])
    cross.to_csv(out_dir / f"{label}_cross_per_seed.csv", index=False)
    cross_mean.to_csv(out_dir / f"{label}_cross_mean.csv", index=False)
    latex_cross(cross_mean, label)
    unstable = mean[(mean.f1_std > 5) | (mean.n_seeds < 3)] if not mean.empty else mean
    if len(unstable):
        print(f"% WARNING {label}: cells with fewer than 3 seeds or seed std above 5 F1:")
        print(unstable.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main_root", type=Path, default=Path("results/generated/xlmr"))
    parser.add_argument("--cross_root", type=Path, default=Path("results/generated/cross_language/xlmr"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/generated/xlmr"))
    parser.add_argument("--compare_root", type=Path, default=None,
                        help="another backbone's main-grid root laid out like --main_root")
    parser.add_argument("--compare_cross_root", type=Path, default=None)
    args = parser.parse_args()
    run(args.main_root, args.cross_root, args.out_dir, "XLM-R")
    if args.compare_root:
        run(args.compare_root, args.compare_cross_root or Path("/nonexistent"), args.out_dir,
            args.compare_root.name)


if __name__ == "__main__":
    main()
