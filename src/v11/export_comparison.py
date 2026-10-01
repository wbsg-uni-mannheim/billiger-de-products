"""Export completed v1.1 results with fixed historical classifier selection."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

root = Path.cwd()
base = root.parent
status = json.loads(Path("campaign_status.json").read_text())
selection = json.loads(Path("comparison_classifier_selection.json").read_text())
assert all(r["old_csv_matches"] for r in selection)
select = {
    (r["model"], r["language"], r["cc"], r["size"], r["test"]): r["classifier"]
    for r in selection
}
tests = {"000": "Seen", "050": "Half-Seen", "100": "Unseen"}
records, controls = [], []


def append(model, table, language, cc, size, test, seed, metrics, source, **extra):
    records.append(
        dict(
            model=model,
            table=table,
            language=language,
            cc=cc,
            size=size,
            test=test,
            seed=seed,
            precision=100 * float(metrics["precision"]),
            recall=100 * float(metrics["recall"]),
            f1=100 * float(metrics["f1"]),
            source=str(source),
            **extra,
        )
    )


def score(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    return dict(
        precision=tp / (tp + fp) if tp + fp else 0,
        recall=tp / (tp + fn) if tp + fn else 0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
    )


for model in ("wordcooc", "magellan"):
    reference = (
        base
        / f"results/generated/cross_language/{model}/de_train_000un_valid_000un_test_050un"
    )
    de_de = next(reference.glob("*_gs_de_de_*.csv"))
    cross_clf = (
        pd.read_csv(de_de, sep="#####", engine="python")
        .groupby("model")
        .mean_valid_score.mean()
        .idxmax()
    )
    for job in status["jobs"]:
        if job.get("model") != model or job["state"] != "COMPLETED":
            continue
        out = (
            root
            / f"results/generated/{model}/{job['mode']}/{job['language']}/{job['family']}-{job['size']}/{job['seed']}"
        )
        assert (out / "complete.json").exists()
        report_folder = (
            "learning-curve"
            if job["mode"] == "main"
            else "de_train_000un_valid_000un_test_050un"
        )
        for source in sorted((out / report_folder).glob("*.csv")):
            frame = pd.read_csv(source, sep="#####", engine="python")
            test = (
                tests[re.search(r"rnd(\d{3})un_gs", source.name).group(1)]
                if job["mode"] == "main"
                else re.search(
                    r"_gs_(de_de|de_en|en_de|en_en|random)_", source.name
                ).group(1)
            )
            cc = job["family"][:2]
            clf = (
                select[model, job["language"], cc, job["size"], test]
                if job["mode"] == "main"
                else cross_clf
            )
            chosen = frame[frame.model == clf]
            assert len(chosen) == 1
            row = chosen.iloc[0]
            paired_source = (
                (
                    base
                    / f"results/generated/{model}/{job['language']}/learning-curve"
                    / source.name
                )
                if job["mode"] == "main"
                else reference / source.name
            )
            paired = {}
            if paired_source.exists():
                previous = pd.read_csv(paired_source, sep="#####", engine="python")
                previous = previous[previous.model == clf].iloc[int(job["seed"]) - 1]
                paired = dict(
                    paired_old_f1=100 * float(previous.f1_test),
                    paired_source=str(paired_source),
                )
            append(
                model,
                "main_grid" if job["mode"] == "main" else "cross_language",
                job["language"],
                cc,
                job["size"],
                test,
                job["seed"],
                dict(
                    precision=row.precision_test, recall=row.recall_test, f1=row.f1_test
                ),
                source,
                classifier=clf,
                validation_score=float(row.mean_valid_score),
                **paired,
            )

for job in status["jobs"]:
    if job["queue"] != "gpu" or job["state"] != "COMPLETED":
        continue
    out = Path(job["output"])
    if job["kind"] == "infer":
        for source in sorted(out.glob("*_metrics.json")):
            if "_valid_" in source.name:
                continue
            test = (
                tests[re.search(r"rnd(\d{3})un_gs", source.name).group(1)]
                if job["mode"] == "main"
                else re.search(
                    r"_gs_(de_de|de_en|en_de|en_en|random)_metrics", source.name
                ).group(1)
            )
            suffix = {
                "Seen": "predict_results.json",
                "Half-Seen": "predict_un050_results.json",
                "Unseen": "predict_un100_results.json",
            }.get(test, f"predict_cross_{test}_results.json")
            paired_source = Path(job["checkpoint"]) / suffix
            previous = json.loads(paired_source.read_text())
            previous_f1 = [v for k, v in previous.items() if k.endswith("_f1")]
            assert len(previous_f1) == 1
            append(
                job["model"],
                "main_grid" if job["mode"] == "main" else "cross_language",
                job["language"],
                job["family"][:2],
                job["size"],
                test,
                job["seed"],
                json.loads(source.read_text()),
                source,
                paired_old_f1=100 * previous_f1[0],
                paired_source=str(paired_source),
            )
    elif job["mode"] == "main":
        for source in out.glob("*_predictions.csv"):
            frame = pd.read_csv(source)
            test = tests[re.search(r"_(\d{3})un_predictions", source.name).group(1)]
            append(
                job["model"],
                "main_grid",
                job["language"],
                job["family"][:2],
                job["size"],
                test,
                job["seed"],
                score(frame.label.tolist(), frame.prediction.tolist()),
                source,
            )

for mode in ("main", "cross"):
    for marker in sorted(
        Path(f"results/generated/gpt/{mode}").glob("*/*/*/complete.json")
    ):
        out = marker.parent
        prompt = out.parent.parent.name
        if mode == "main":
            language, family, condition = out.parent.name.split("_")
            test = tests[condition]
            cc = family[:2]
            size = ""
            seed = 0
            original_requests = list(
                map(json.loads, (out / "original_requests.jsonl").open())
            )
            original_metadata = json.loads(
                (out / "original_requests_meta.json").read_text()
            )
            cache_file = (
                Path("recovered_gpt_history") / prompt / (out.parent.name + ".jsonl")
            )
            cache = {
                r["request_sha256"]: r["answer"]
                for r in map(json.loads, cache_file.open())
            }
            labels, predictions, missing = [], [], []
            for req in original_requests:
                key = hashlib.sha256(
                    json.dumps(req["body"], sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                if key not in cache:
                    missing.append(req["custom_id"])
                    continue
                answer = cache[key].strip().lower().rstrip(".,")
                assert answer in ("ja", "nein", "yes", "no")
                labels.append(int(original_metadata[req["custom_id"]]["label"]))
                predictions.append(int(answer in ("ja", "yes")))
            controls.append(
                dict(
                    prompt=prompt,
                    language=language,
                    cc=cc,
                    test=test,
                    pairs=len(original_requests),
                    missing=len(missing),
                    missing_ids=missing,
                    **score(labels, predictions),
                )
            )
        else:
            language = "de"
            cc = "80"
            size = "large"
            test = out.parent.name
            seed = int(out.name)
        append(
            "gpt-5.2-" + prompt.replace("_", "-"),
            "main_grid" if mode == "main" else "cross_language",
            language,
            cc,
            size,
            test,
            seed,
            json.loads(marker.read_text()),
            marker,
        )

Path("comparison_export.json").write_text(
    json.dumps(
        dict(
            timestamp_utc=status["timestamp_utc"],
            rows=records,
            gpt_v10_controls=controls,
        ),
        indent=2,
    )
    + "\n"
)
print(
    "Exported", len(records), "per-run results and", len(controls), "GPT v1.0 controls"
)
