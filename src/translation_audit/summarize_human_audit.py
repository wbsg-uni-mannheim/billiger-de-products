"""Summarize the human translation audit: matching-relevant errors per stratum and the weighted rate.

Run from the repository root: python src/translation_audit/summarize_human_audit.py
"""
import collections
import csv
from pathlib import Path

ANNOTATION = Path("reports/translation_audit/manual_annotation")
POPULATION = {"flagged": 75, "unanimous_clean": 2458, "other": 466}
SAMPLE = {"flagged": 75, "unanimous_clean": 100, "other": 50}

key = {r["annotation_id"]: r["stratum"]
       for r in csv.DictReader(open(ANNOTATION / "analysis_only/sampling_key.csv", encoding="utf-8-sig"))}
tally = collections.Counter()
for path in sorted(ANNOTATION.glob("translations_0*.csv")):
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        tally[key[row["annotation_id"]], row["human_verdict"].strip()] += 1

print(f"{'stratum':16s} {'population':>10s} {'annotated':>9s} {'acceptable':>10s} {'minor':>5s} {'relevant':>8s}")
for s in POPULATION:
    counts = [tally[s, v] for v in ("Ja okay", "Nein nicht relevant", "Nein relevant")]
    assert sum(counts) == SAMPLE[s], (s, counts)
    print(f"{s:16s} {POPULATION[s]:10d} {SAMPLE[s]:9d} {counts[0]:10d} {counts[1]:5d} {counts[2]:8d}")
rate = sum(POPULATION[s] * tally[s, "Nein relevant"] / SAMPLE[s] for s in POPULATION) / sum(POPULATION.values())
print(f"population-weighted matching-relevant error rate: {100 * rate:.2f} %")
