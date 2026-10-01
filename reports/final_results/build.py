"""Build the per-seed results on benchmark v1.1 and the per-cell summary.

Usage (from the repository root):
    python reports/final_results/build.py <prediction dir> <server export json> <collapse audit json>

<prediction dir> holds the Ditto/HierGAT cross-language prediction CSVs (cross/de/80cc20-large/<seed>/)
and the restart outputs (restarts_v11_*/.../restart_decision.json plus prediction CSVs), synced from
v11-reruns-20260906/results/generated on the cluster. <server export json> is the per-run export written
by src/v11/export_comparison.py. <collapse audit json> is the validation audit of all Ditto/HierGAT runs.

Collapse rule: a Ditto/HierGAT run is collapsed if the Seen-validation F1 of its selected checkpoint is
less than 0.10 above the F1 of predicting every pair as a match. A collapsed run is replaced by an
accepted restart with a new seed where one exists, otherwise it is excluded and the cell averages the
remaining seeds.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
PRED_DIR, EXPORT, AUDIT = map(Path, sys.argv[1:4])
MARGIN = 0.10
COLS = ["model", "table", "language", "cc", "size", "test", "seed", "precision", "recall", "f1"]
KEYS = ["model", "table", "language", "cc", "size", "test"]
TESTS = {"000un": "Seen", "050un": "Half-Seen", "100un": "Unseen"}


def score(frame):
    y, p = frame.label, frame.prediction
    tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
    return dict(
        precision=100 * tp / (tp + fp) if tp + fp else 0.0,
        recall=100 * tp / (tp + fn) if tp + fn else 0.0,
        f1=100 * 2 * tp / (2 * tp + fp + fn) if tp + fn else 0.0,
    )


def prediction_rows(files, model, table, language, cc, size, seed):
    rows = []
    for path in sorted(files):
        m = re.search(r"_(000un|050un|100un|cross_(?:de_de|de_en|en_de|en_en|random))_predictions", path.name)
        test = TESTS.get(m.group(1), m.group(1).removeprefix("cross_"))
        if table == "cross_language" and test in TESTS.values():
            continue
        rows.append(dict(model=model, table=table, language=language, cc=cc, size=size, test=test, seed=seed,
                         **score(pd.read_csv(path))))
    return rows


def slot(model, mode, language, family, size, seed):
    return (model, "main_grid" if mode == "main" else "cross_language", language, int(family[:2]), size, int(seed))


rows = pd.DataFrame(json.loads(EXPORT.read_text())["rows"])
rows["cc"] = rows.cc.astype(int)
rows["seed"] = rows.seed.astype(int)
rows["size"] = rows["size"].fillna("")
rows = rows[COLS]

# Ditto/HierGAT cross-language runs are not part of the server export.
extra = []
for folder in sorted((PRED_DIR / "cross/de/80cc20-large").iterdir()):
    for model, marker in (("ditto", "_da=del"), ("hiergat", "_lr=")):
        extra += prediction_rows(folder.glob(f"*{marker}*_predictions.csv"), model, "cross_language", "de", 80, "large",
                                 int(folder.name))
rows = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)

# Collapsed runs, decided on Seen validation only.
audit = json.loads(AUDIT.read_text())
collapsed = {slot(r["model"], r["mode"], r["language"], r["family"], r["size"], r["seed"]): r for r in audit
             if r["best_validation_f1"] < r["all_positive_validation_f1"] + MARGIN}

restarts, log = [], []
for decision_file in sorted(PRED_DIR.glob("restarts_v11_*/**/restart_decision.json")):
    d = json.loads(decision_file.read_text())
    _, _, model, mode, language, family, size, original_seed, _ = d["name"].split("_")
    key = slot(model, mode, language, family, size, original_seed)
    if key not in collapsed or not d["accepted"]:
        continue
    new = prediction_rows(decision_file.parent.glob("*_predictions.csv"), *key[:5], d["replacement_seed"])
    assert new, decision_file
    restarts += new
    log.append(dict(zip(["model", "table", "language", "cc", "size", "seed"], key),
                    validation_f1=collapsed[key]["best_validation_f1"],
                    all_positive_validation_f1=collapsed[key]["all_positive_validation_f1"],
                    action=f"replaced by seed {d['replacement_seed']} (validation F1 {d['validation_f1']:.3f})"))
replaced = {(r["model"], r["table"], r["language"], r["cc"], r["size"], r["seed"]) for r in log}
for key, r in sorted(collapsed.items()):
    if key not in replaced:
        log.append(dict(zip(["model", "table", "language", "cc", "size", "seed"], key),
                        validation_f1=r["best_validation_f1"], all_positive_validation_f1=r["all_positive_validation_f1"],
                        action="excluded"))

drop = rows.apply(lambda r: (r.model, r.table, r.language, r.cc, r["size"], r.seed) in collapsed, axis=1)
final = pd.concat([rows[~drop], pd.DataFrame(restarts)], ignore_index=True)
assert not final.duplicated(KEYS + ["seed"]).any()

for model, frame in final.groupby("model"):
    frame[COLS].round(4).sort_values(COLS[:7]).to_csv(OUT / f"{model}.csv", index=False)
pd.DataFrame(log).round(4).sort_values(["model", "table", "language", "cc", "size", "seed"]).to_csv(
    OUT / "collapsed_runs.csv", index=False)
summary = (final.groupby(KEYS).agg(precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
                                   f1_std=("f1", "std"), seeds=("f1", "size")).reset_index())
summary.round(2).to_csv(OUT / "summary.csv", index=False)
print(len(final), "runs,", len(summary), "cells,", ((summary.seeds < 3) & ~summary.model.str.startswith("gpt")).sum(), "supervised cells with fewer than three seeds")
print(pd.DataFrame(log).action.value_counts().to_string())
