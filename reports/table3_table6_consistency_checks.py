"""Recompute every measured figure quoted in table3_table6_consistency.md.

Run from the repository root with an interpreter that has pandas available,
e.g. the environment the cross-language jobs use:

    /home/aasteine/miniconda3/envs/ditto-modern/bin/python \
        reports/table3_table6_consistency_checks.py
"""

import ast
import glob
import json
import os
import statistics
from collections import defaultdict

import pandas as pd


LEGACY = "results/generated/cross_language_legacy_050un_valid"


def pct(value):
    return round(value * 100, 2)


def offer_pair_keys(pairs):
    return {
        tuple(sorted((str(left), str(right))))
        for left, right in zip(pairs["id_left"], pairs["id_right"])
    }


def splits():
    print("== validation and test splits ==")
    train = pd.read_pickle(
        "data/processed/training-sets/preprocessed_products80cc20rnd000un_train_large.pkl.gz"
    )
    train_pairs = offer_pair_keys(train)
    train_products = set(train["product_id_left"]) | set(train["product_id_right"])

    for name, path in (
        ("train", "data/processed/training-sets/preprocessed_products80cc20rnd000un_train_large.pkl.gz"),
        ("valid_000un", "data/processed/validation-sets/preprocessed_products80cc20rnd000un_valid_large.pkl.gz"),
        ("valid_050un (released)", "data/processed/validation-sets/preprocessed_products80cc20rnd050un_valid_large.pkl.gz"),
        ("valid_050un (cross, dedup)", "data/processed_cross_language/validation-sets/preprocessed_products80cc20rnd050un_valid_large.pkl.gz"),
        ("gs_050un (test)", "data/processed/gold-standards_adjusted/preprocessed_products80cc20rnd050un_gs.pkl.gz"),
    ):
        if not os.path.exists(path):
            print(f"{name:28s} MISSING {path}")
            continue
        frame = pd.read_pickle(path)
        products = set(frame["product_id_left"]) | set(frame["product_id_right"])
        print(
            f"{name:28s} rows={len(frame):5d} pos={int(frame['label'].sum()):5d} "
            f"seen_product_share={len(train_products & products) / len(products):.4f} "
            f"offer_pair_overlap_with_train={len(train_pairs & offer_pair_keys(frame))}"
        )


def main_cells():
    print("\n== Table 3 cells, 80cc20rnd / large / Half-Seen ==")

    root = "results/generated/roberta_bs32_full/de/products80cc20rnd000un-large"
    values = [
        json.load(open(f"{root}/{seed}/predict_un050_results.json"))["predict_un050_f1"]
        for seed in "012"
    ]
    print(f"RoBERTa  (bs=32, current)  {[pct(v) for v in values]} mean={pct(statistics.mean(values))}")

    root = ("src/models/transformer_bert_confidence/reports/baseline/"
            "products80cc20rnd000un-large-all1024-5e-05-roberta-base_adjusted")
    values = [
        json.load(open(f"{root}/{seed}/predict_un050_results.json"))["predict_un050_f1"]
        for seed in "012"
    ]
    print(f"RoBERTa  (bs=1024, website) {[pct(v) for v in values]} mean={pct(statistics.mean(values))}")

    root = ("src/models/r-supCon/reports/contrastive-ft-siamese/"
            "products80cc20rnd000un-large-all1024-5e-05-0.07-False-roberta-base_adjusted")
    values = [
        json.load(open(f"{root}/{seed}/predict_un050_results.json"))["predict_un050_f1"]
        for seed in "012"
    ]
    print(f"R-SupCon {[pct(v) for v in values]} mean={pct(statistics.mean(values))}")

    values = []
    for path in sorted(glob.glob("src/models/hiergat/output/final_large_80cc20rnd000un_*_adjusted.txt")):
        values.append(ast.literal_eval(open(path).read().strip())["best_test_f1_050"])
    print(f"HierGAT  {[pct(v) for v in values]} mean={pct(statistics.mean(values))} (published cell: 65.40)")

    ditto = {}
    for path in sorted(glob.glob("src/models/ditto/output/final_large_80cc20rnd000un_*_adjusted_testset.txt")):
        run_id = path.split("id=")[1].split("_")[0]
        ditto[run_id] = ast.literal_eval(open(path).read().strip())["best_f1_050"]
    print("Ditto    per run_id " + ", ".join(f"{k}={pct(v)}" for k, v in sorted(ditto.items())))
    print(f"Ditto    seeds 0,1,2 mean={pct(statistics.mean(ditto[i] for i in '012'))}")
    print(f"Ditto    ids 0,3,4  mean={pct(statistics.mean(ditto[i] for i in '034'))} (published cell: 64.10)")

    for name, path in (
        ("WordCooc", "src/models/wordcooc/model_output_adjusted_ts/reports/wordcooc_adjusted/learning-curve_adjusted/"
                     "preprocessed_products80cc20rnd000un_train_large_wordcooc_"
                     "preprocessed_products80cc20rnd000un_train_large_wordcooc_"
                     "preprocessed_products80cc20rnd050un_gs.csv"),
        ("Magellan", "src/models/magellan/model_output_adjusted_ts/reports/learning-curve_adjusted/"
                     "preprocessed_products80cc20rnd000un_train_large_magellan_pairs_formatted_"
                     "preprocessed_products80cc20rnd050un_gs_magellan_pairs_formatted.csv"),
    ):
        print(f"{name} (main): " + summarize_table(path))


def summarize_table(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    header = lines[0].split("#####")
    grouped = defaultdict(list)
    for line in lines[1:]:
        values = dict(zip(header, line.split("#####")))
        grouped[values["model"]].append(float(values["f1_test"]))
    ranked = sorted(grouped.items(), key=lambda item: -statistics.mean(item[1]))
    parts = [f"rows={len(lines) - 1}"]
    parts += [f"{name} n={len(v)} mean={pct(statistics.mean(v))}" for name, v in ranked]
    return " | ".join(parts)


def legacy_cross_language():
    if not os.path.isdir(LEGACY):
        print(f"\n({LEGACY} not present, skipping legacy cross-language checks)")
        return
    print("\n== old Table 6 DE-DE column (legacy 050un-validation run) ==")

    for name, root, key in (
        ("RoBERTa", f"{LEGACY}/roberta/80cc20-large", "predict_cross_de_de_f1"),
        ("R-SupCon", f"{LEGACY}/r-supcon/80cc20-large", "predict_cross_de_de_f1"),
    ):
        values = []
        for seed in "012":
            payload = json.load(open(f"{root}/{seed}/predict_cross_de_de_results.json"))
            values.append(next(v for k, v in payload.items() if k.endswith("f1")))
        print(f"{name} {[pct(v) for v in values]} mean={pct(statistics.mean(values))}")

    for name, pattern, key in (
        ("HierGAT", f"{LEGACY}/hiergat/*.txt", "best_test_f1_cross_de_de"),
        ("Ditto", f"{LEGACY}/ditto/*.txt", "best_f1_cross_de_de"),
    ):
        values = {}
        for path in sorted(glob.glob(pattern)):
            run_id = path.split("id=")[1].split("_")[0]
            values[run_id] = ast.literal_eval(open(path).read().strip())[key]
        print(f"{name} per run_id " + ", ".join(f"{k}={pct(v)}" for k, v in sorted(values.items())))

    for name, pattern in (
        ("WordCooc", f"{LEGACY}/wordcooc/*/*de_de*.csv"),
        ("Magellan", f"{LEGACY}/magellan/*/*de_de*.csv"),
    ):
        for path in sorted(glob.glob(pattern)):
            print(f"{name} (cross): " + summarize_table(path))


def selected_checkpoints():
    print("\n== selected checkpoints, RoBERTa ==")
    for name, root in (
        ("cross-language (050un valid, bs=1024)", f"{LEGACY}/roberta/80cc20-large"),
        ("main grid (000un valid, bs=32)", "results/generated/roberta_bs32_full/de/products80cc20rnd000un-large"),
    ):
        if not os.path.isdir(root):
            print(f"{name}: MISSING {root}")
            continue
        parts = []
        for seed in "012":
            state = f"{root}/{seed}/trainer_state.json"
            if not os.path.exists(state):
                parts.append(f"{seed}=?")
                continue
            payload = json.load(open(state))
            parts.append(
                f"{seed}={os.path.basename(str(payload.get('best_model_checkpoint')))}"
                f"@{round(payload.get('best_metric') or 0, 4)}"
            )
        print(f"{name}: " + " ".join(parts))


if __name__ == "__main__":
    splits()
    main_cells()
    legacy_cross_language()
    selected_checkpoints()
