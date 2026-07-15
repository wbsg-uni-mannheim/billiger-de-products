"""Collect F1 scores emitted by all benchmark implementations into one CSV."""

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def context(path):
    text = path.as_posix().lower()
    model = next(
        (name for name in ("wordcooc", "magellan", "ditto", "hiergat", "r-supcon", "roberta", "gpt") if name in text),
        "unknown",
    )
    language = "en" if "/en/" in text or "_en/" in text or "english" in text else "de"
    return model, language


def rows_from_json(path):
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(values, dict):
        return []
    return [(key, value, "") for key, value in values.items() if "f1" in key.lower() and isinstance(value, (int, float))]


def rows_from_text(path):
    text = path.read_text(encoding="utf-8")
    try:
        values = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        values = None
    if isinstance(values, dict):
        return [(key, value, "") for key, value in values.items() if "f1" in key.lower() and isinstance(value, (int, float))]
    match = re.search(r"F1 Score:\s*([0-9.]+)", text)
    return [("f1", float(match.group(1)), "")] if match else []


def rows_from_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("#####")
    rows = []
    for line in lines[1:]:
        values = line.split("#####")
        row = dict(zip(header, values))
        if row.get("f1_test"):
            rows.append(("f1_test", float(row["f1_test"]), row.get("model", "")))
    return rows


def collect(root):
    collected = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            metrics = rows_from_json(path)
        elif path.suffix == ".txt":
            metrics = rows_from_text(path)
        elif path.suffix == ".csv" and "#####" in path.read_text(encoding="utf-8", errors="ignore")[:1000]:
            metrics = rows_from_table(path)
        else:
            continue
        model, language = context(path)
        for metric, f1, classifier in metrics:
            collected.append(
                {
                    "model": model,
                    "language": language,
                    "metric": metric,
                    "classifier": classifier,
                    "f1": f1,
                    "source_file": path.as_posix(),
                }
            )
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/generated"))
    parser.add_argument("--output", type=Path, default=Path("results/generated/metrics.csv"))
    args = parser.parse_args()

    rows = collect(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "language", "metric", "classifier", "f1", "source_file"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metrics to {args.output}")


if __name__ == "__main__":
    main()
