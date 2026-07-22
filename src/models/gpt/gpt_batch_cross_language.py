"""Evaluate GPT-5.2 zero-shot on the aligned cross-language test sets."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.cross_language.common import DISPLAY_NAMES, VARIANTS, raw_path


SIMPLE_PROMPT = (
    "Beziehen sich diese beiden Produktbeschreibungen auf dasselbe reale Produkt?\n"
    "Antworte nur mit Ja oder Nein.\n"
    "Produkt 1: {left}\n"
    "Produkt 2: {right}"
)

RULE_GUIDED_PROMPT = (
    "Du bist ein Experte für Produktabgleich. Deine Aufgabe ist zu entscheiden, "
    "ob sich zwei Produktdatensätze auf das EXAKT gleiche Produkt beziehen "
    "(gleiche GTIN/SKU).\n"
    "Analysiere die bereitgestellten Datensätze sorgfältig und gib deine "
    "Entscheidung strikt als Ja oder Nein zurück.\n\n"
    "KRITISCH: Produktvarianten sind KEINE Übereinstimmungen. Unterschiedliche "
    "Größen, Farben, Konfigurationen oder Verpackungsmengen sind UNTERSCHIEDLICHE "
    "Produkte mit unterschiedlichen GTINs.\n\n"
    "Richtlinien:\n"
    "- Ja NUR, wenn sich die Datensätze auf exakt dasselbe Produkt beziehen, "
    "das dieselbe GTIN/denselben Barcode hätte\n"
    "- Nein, wenn es sich um Varianten derselben Produktlinie handelt "
    "(unterschiedliche Größe, Farbe, Kapazität usw.)\n"
    "- Nein bei widersprüchlichen zentralen Identifikationsmerkmalen "
    "(Modellnummern, Abmessungen, Kapazität, Farbe und Konfiguration)\n"
    "- Fehlende Attribute allein sind KEIN Widerspruch.\n"
    "- Eine Übereinstimmung erfordert positive Evidenz der Gleichheit\n"
    "- Antworte AUSSCHLIESSLICH mit Ja oder Nein.\n\n"
    "Produkt 1: {left}\n"
    "Produkt 2: {right}\n"
)


def serialize_record(record, side):
    labels = {
        "brand": "Marke",
        "name": "Name",
        "price": "Preis",
        "desc": "Beschreibung",
    }
    parts = []
    for attribute, label in labels.items():
        value = record.get(f"{attribute}_{side}")
        if value is not None:
            parts.append(f"{label}: {value}")
    return " ".join(parts).replace("/", " ")


def build_batch_file(variant, prompt_name, model, batch_path):
    template = SIMPLE_PROMPT if prompt_name == "simple" else RULE_GUIDED_PROMPT
    metadata = {}
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_json(raw_path(variant), lines=True, compression="gzip")
    with batch_path.open("w", encoding="utf-8") as output:
        for record in pairs.to_dict(orient="records"):
            pair_id = str(record["pair_id"])
            left = serialize_record(record, "left")
            right = serialize_record(record, "right")
            request = {
                "custom_id": pair_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": template.format(left=left, right=right),
                        }
                    ],
                },
            }
            output.write(json.dumps(request, ensure_ascii=False) + "\n")
            metadata[pair_id] = {
                "label": int(record["label"]),
                "is_hard_negative": bool(record["is_hard_negative"]),
                "language_left": record["language_left"],
                "language_right": record["language_right"],
            }

    metadata_path = batch_path.with_name(f"{batch_path.stem}_meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def run_batch(client, batch_path, result_path):
    uploaded = client.files.create(file=batch_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch ID: {batch.id}")

    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"Status: {batch.status}")
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status {batch.status}")
        time.sleep(60)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(batch.output_file_id)
    result_path.write_bytes(content.read())


def parse_answer(answer):
    normalized = answer.strip().lower()
    if normalized in ("ja", "1") or "ja" in normalized:
        return 1
    if normalized in ("nein", "0") or "nein" in normalized:
        return 0
    return -1


def collect_results(result_path, metadata_path, output_dir):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []

    with result_path.open(encoding="utf-8") as results:
        for line in results:
            response = json.loads(line)
            pair_id = response["custom_id"]
            answer = response["response"]["body"]["choices"][0]["message"]["content"]
            rows.append(
                {
                    "pair_id": pair_id,
                    "answer": answer,
                    "prediction": parse_answer(answer),
                    **metadata[pair_id],
                }
            )

    frame = pd.DataFrame(rows)
    valid = frame[frame["prediction"] != -1]
    metrics = {
        "accuracy": accuracy_score(valid["label"], valid["prediction"]),
        "precision": precision_score(valid["label"], valid["prediction"], zero_division=0),
        "recall": recall_score(valid["label"], valid["prediction"], zero_division=0),
        "f1": f1_score(valid["label"], valid["prediction"], zero_division=0),
        "invalid_answers": int((frame["prediction"] == -1).sum()),
        "pairs": len(frame),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--prompt", choices=("simple", "rule_guided"), required=True)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Repetition index. The requests are sampled with the API default "
        "temperature, so repeated runs differ. Runs > 1 are written to separate "
        "files so earlier runs are never overwritten.",
    )
    args = parser.parse_args()

    suffix = "" if args.run <= 1 else f"_run{args.run}"
    batch_dir = Path("data/batch_inputs/cross_language/gpt") / args.model / args.prompt
    result_dir = Path("data/batch_results/cross_language/gpt") / args.model / args.prompt
    output_dir = (
        Path("results/generated/cross_language/gpt")
        / args.model
        / args.prompt
        / f"{args.variant}{suffix}"
    )
    batch_path = batch_dir / f"{args.variant}{suffix}.jsonl"
    metadata_path = build_batch_file(
        args.variant,
        args.prompt,
        args.model,
        batch_path,
    )
    print(f"Prepared {DISPLAY_NAMES[args.variant]} batch input: {batch_path}")

    if args.prepare_only:
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --prepare-only is used")
    result_path = result_dir / f"{args.variant}{suffix}.jsonl"
    run_batch(OpenAI(api_key=api_key), batch_path, result_path)
    collect_results(result_path, metadata_path, output_dir)


if __name__ == "__main__":
    main()
