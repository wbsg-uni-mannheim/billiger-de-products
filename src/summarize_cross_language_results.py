"""Collect cross-language precision, recall, and F1 results into one CSV."""

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path

from src.cross_language.common import VARIANTS


MODEL_NAMES = ("wordcooc", "magellan", "roberta", "r-supcon", "ditto", "hiergat", "gpt")


def model_from_path(path):
    # Match the model directory exactly: Ditto/HierGAT result *files* contain
    # "lm=roberta" in their name, which would otherwise be attributed to RoBERTa.
    parts = {part.lower() for part in path.parts}
    for model in MODEL_NAMES:
        if model in parts:
            return model
    text = path.as_posix().lower()
    return next(model for model in MODEL_NAMES if model in text)


def variant_from_text(text):
    return next((variant for variant in VARIANTS if variant in text), None)


def rows_from_json(path):
    # "all_results.json" is the Hugging Face aggregate and repeats every
    # "predict_cross_<variant>_f1" key that already has its own result file,
    # which would count each seed twice.
    if path.name == "all_results.json":
        return []

    values = json.loads(path.read_text(encoding="utf-8"))
    model = model_from_path(path)
    variant = variant_from_text(path.as_posix())

    if model == "gpt" and variant and "f1" in values:
        return [
            {
                "model": model,
                "classifier": path.parent.parent.name,
                "seed": "",
                "variant": variant,
                "precision": values.get("precision", ""),
                "recall": values.get("recall", ""),
                "f1": values["f1"],
                "source_file": path.as_posix(),
            }
        ]

    rows = []
    for key, f1 in values.items():
        if "cross_" not in key or not key.endswith("_f1"):
            continue
        key_variant = variant_from_text(key)
        prefix = key.removesuffix("_f1")
        rows.append(
            {
                "model": model,
                "classifier": "",
                "seed": path.parent.name if path.parent.name in ("0", "1", "2") else "",
                "variant": key_variant,
                "precision": values.get(f"{prefix}_precision", ""),
                "recall": values.get(f"{prefix}_recall", ""),
                "f1": f1,
                "source_file": path.as_posix(),
            }
        )
    return rows


def rows_from_text(path):
    try:
        values = ast.literal_eval(path.read_text(encoding="utf-8").strip())
    except (SyntaxError, ValueError):
        return []
    if not isinstance(values, dict):
        return []

    seed_match = re.search(r"_id=(\d+)", path.name)
    rows = []
    for key, f1 in values.items():
        if "f1_cross_" not in key or not isinstance(f1, (int, float)):
            continue
        precision_key = key.replace("f1_cross_", "precision_cross_")
        recall_key = key.replace("f1_cross_", "recall_cross_")
        rows.append(
            {
                "model": model_from_path(path),
                "classifier": "",
                "seed": seed_match.group(1) if seed_match else "",
                "variant": variant_from_text(key),
                "precision": values.get(precision_key, ""),
                "recall": values.get(recall_key, ""),
                "f1": f1,
                "source_file": path.as_posix(),
            }
        )
    return rows


def rows_from_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or "#####" not in lines[0]:
        return []
    header = lines[0].split("#####")
    variant = variant_from_text(path.name)
    counters = {}
    rows = []

    for line in lines[1:]:
        values = dict(zip(header, line.split("#####")))
        classifier = values.get("model", "")
        counters[classifier] = counters.get(classifier, 0)
        seed = counters[classifier]
        counters[classifier] += 1
        rows.append(
            {
                "model": model_from_path(path),
                "classifier": classifier,
                "seed": seed,
                "variant": variant,
                "precision": values.get("precision_test", ""),
                "recall": values.get("recall_test", ""),
                "f1": values.get("f1_test", ""),
                "source_file": path.as_posix(),
            }
        )
    return rows


def collect(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                rows.extend(rows_from_json(path))
            elif path.suffix == ".txt":
                rows.extend(rows_from_text(path))
            elif path.suffix == ".csv":
                rows.extend(rows_from_table(path))
        except (StopIteration, json.JSONDecodeError):
            continue
    return [row for row in rows if row["variant"] in VARIANTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/generated/cross_language"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generated/cross_language/metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/generated/cross_language/summary.csv"),
    )
    args = parser.parse_args()

    rows = collect(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "model",
                "classifier",
                "seed",
                "variant",
                "precision",
                "recall",
                "f1",
                "source_file",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["model"], row["classifier"], row["variant"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (model, classifier, variant), group in sorted(grouped.items()):
        summary = {
            "model": model,
            "classifier": classifier,
            "variant": variant,
            "runs": len(group),
        }
        for metric in ("precision", "recall", "f1"):
            values = [
                float(row[metric])
                for row in group
                if row[metric] not in ("", None)
            ]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else ""
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "model",
                "classifier",
                "variant",
                "runs",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(rows)} cross-language metrics to {args.output}")
    print(f"Wrote {len(summary_rows)} summaries to {args.summary_output}")


if __name__ == "__main__":
    main()
