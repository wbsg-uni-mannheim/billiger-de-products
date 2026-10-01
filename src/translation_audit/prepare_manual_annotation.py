"""Create three blind annotation CSVs and a separate sampling key.

Run from the repository root. Existing output directories are never overwritten.
CSV export uses the standard library because the bundled artifact-tool exposes
CSV import but no documented CSV export operation.
"""

import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("reports/translation_audit")
OUT = ROOT / "manual_annotation"
SEED = 20260905
TEXT_FIELDS = ("de_name", "en_name", "de_desc", "en_desc")
FIELDS = ("annotation_id", "offer_id", *TEXT_FIELDS, "human_verdict", "human_note")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite possible annotations: {OUT}")
    verdicts = read_csv(ROOT / "audit_verdicts.csv")
    flagged = read_csv(ROOT / "flagged_for_review.csv")
    clean = read_csv(ROOT / "calibration_sample.csv")
    models = [key for key in verdicts[0] if key not in {"id", "majority", "issues"}]
    strata = {"flagged": [], "unanimous_clean": [], "other": []}
    for row in verdicts:
        values = [row[model] for model in models]
        group = ("flagged" if "critical error" in values else
                 "unanimous_clean" if set(values) == {"error-free"} else "other")
        strata[group].append(row)
    assert {r["id"] for r in flagged} == {r["id"] for r in strata["flagged"]}
    assert len(flagged) == 75 and len(clean) == 100
    assert {r["id"] for r in clean} <= {r["id"] for r in strata["unanimous_clean"]}
    other = random.Random(SEED).sample(sorted(strata["other"], key=lambda r: r["id"]), 50)
    selected = [(r, "flagged") for r in flagged]
    selected += [(r, "unanimous_clean") for r in clean]
    selected += [(r, "other") for r in other]
    assert len({r["id"] for r, _ in selected}) == 225
    offers = {r["id"]: {} for r, _ in selected}
    for language in ("de", "en"):
        for path in sorted(Path(f"data/solute_{language}/gold-standards_adjusted").glob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    pair = json.loads(line)
                    for side in ("left", "right"):
                        offer_id = str(pair[f"id_{side}"])
                        if offer_id not in offers:
                            continue
                        for field in ("name", "desc"):
                            value = pair[f"{field}_{side}"]
                            text = "" if value is None else str(value)
                            key = f"{language}_{field}"
                            previous = offers[offer_id].setdefault(key, text)
                            assert previous == text, (offer_id, key, path)
    for row in flagged + clean:
        assert all(row[field] == offers[row["id"]][field] for field in TEXT_FIELDS)
    assert all(set(text) == set(TEXT_FIELDS) for text in offers.values())
    random.Random(SEED + 1).shuffle(selected)
    sample_counts = Counter(group for _, group in selected)
    blind, keys = [], []
    by_id = {row["id"]: row for row in verdicts}
    for index, (row, group) in enumerate(selected, 1):
        annotation_id = f"TA{index:04d}"
        blind.append(dict(zip(FIELDS, [annotation_id, row["id"],
            *(offers[row["id"]][field] for field in TEXT_FIELDS), "", ""])))
        keys.append({"annotation_id": annotation_id, "file": f"translations_{(index - 1) // 75 + 1:02d}.csv",
                     "stratum": group, "population_n": len(strata[group]),
                     "sample_n": sample_counts[group], **by_id[row["id"]]})
    OUT.mkdir()
    private = OUT / "analysis_only"
    private.mkdir()
    for index in range(3):
        path = OUT / f"translations_{index + 1:02d}.csv"
        rows = blind[index * 75:(index + 1) * 75]
        write_csv(path, rows)
        assert read_csv(path) == rows, path
    write_csv(private / "sampling_key.csv", keys)
    manifest = {"sample_seed_other": SEED, "shuffle_seed": SEED + 1,
                "population": "2999 test offers with valid judgments from all three models",
                "excluded_incomplete_audit_offer": "54158093151",
                "strata": {group: {"population": len(rows), "sample": sample_counts[group]}
                           for group, rows in strata.items()},
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("audit_verdicts.csv", "flagged_for_review.csv", "calibration_sample.csv")}}
    (private / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": 3, "offers": len(blind), "strata": manifest["strata"]}, indent=2))


if __name__ == "__main__":
    main()
