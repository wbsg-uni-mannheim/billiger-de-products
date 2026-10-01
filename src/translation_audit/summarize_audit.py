"""Aggregate the OpenRouter audit: majority verdict per offer, agreement between the models,
critical-error rate with a Wilson interval, and the two lists for manual review.

Writes to reports/translation_audit/:
  audit_summary.md          counts, rates, Fleiss' kappa, per-model verdict distribution
  audit_verdicts.csv        one row per offer with every model's verdict and the majority
  flagged_for_review.csv    offers with a majority "critical error" or without a majority
  calibration_sample.csv    100 random offers all models call error-free, for the human check

    python src/translation_audit/summarize_audit.py [--sample 100] [--seed 0]
"""
import argparse, csv, glob, json, math, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from offers import aligned_offers

IN = "results/generated/translation_audit"; OUT = "reports/translation_audit"
VERDICTS = ("error-free", "minor error", "critical error")


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fleiss_kappa(rows):
    """rows: list of Counter(verdict -> count) with the same total per row."""
    n = len(rows); k = rows[0].total() if hasattr(rows[0], "total") else sum(rows[0].values())
    if n == 0 or k < 2: return float("nan")
    p_j = {v: sum(r[v] for r in rows) / (n * k) for v in VERDICTS}
    p_i = [(sum(c * c for c in r.values()) - k) / (k * (k - 1)) for r in rows]
    p_bar = sum(p_i) / n; p_e = sum(p * p for p in p_j.values())
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--sample", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{IN}/*.jsonl")); assert files, f"no results in {IN}"
    models = [os.path.basename(f)[:-6].replace("__", "/") for f in files]
    verdict = {m: {} for m in models}; issue = {m: {} for m in models}
    for m, f in zip(models, files):
        with open(f, encoding="utf-8") as h:
            for line in h:
                r = json.loads(line); p = r.get("parsed")
                if p and p.get("verdict") in VERDICTS:
                    verdict[m][r["id"]] = p["verdict"]; issue[m][r["id"]] = p.get("issue", "")
    ids = sorted(set.intersection(*(set(v) for v in verdict.values())))
    offers = aligned_offers()
    rows, counters, flagged, unanimous_ok = [], [], [], []
    for i in ids:
        vs = [verdict[m][i] for m in models]; c = Counter(vs); top, n_top = c.most_common(1)[0]
        majority = top if n_top >= 2 else "no majority"
        rows.append({"id": i, **{m: verdict[m][i] for m in models}, "majority": majority, "issues": " | ".join(f"{m}: {issue[m][i]}" for m in models if issue[m][i])})
        counters.append(c)
        if majority in ("critical error", "no majority") or "critical error" in vs:
            flagged.append(rows[-1])
        if all(v == "error-free" for v in vs):
            unanimous_ok.append(rows[-1])
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/audit_verdicts.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def enrich(r):
        o = offers[r["id"]]; return {**r, "de_name": o["de"]["name"], "en_name": o["en"]["name"], "de_desc": o["de"]["desc"], "en_desc": o["en"]["desc"], "human_verdict": "", "human_note": ""}
    for name, sel in (("flagged_for_review.csv", flagged), ("calibration_sample.csv", random.Random(a.seed).sample(unanimous_ok, min(a.sample, len(unanimous_ok))))):
        with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as h:
            rs = [enrich(r) for r in sel]
            if rs: w = csv.DictWriter(h, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    n = len(ids); maj = Counter(r["majority"] for r in rows); crit = maj.get("critical error", 0); lo, hi = wilson(crit, n)
    any_crit = sum(1 for r in rows if "critical error" in [r[m] for m in models])
    unanimous = sum(1 for c in counters if len(c) == 1)
    lines = [f"# Translation audit summary", "", f"offers judged by all {len(models)} models: {n}", f"models: {', '.join(models)}", "",
             "## Majority verdict", *(f"- {v}: {maj.get(v, 0)} ({100*maj.get(v,0)/n:.2f} %)" for v in VERDICTS + ("no majority",)), "",
             f"critical-error rate (majority): {100*crit/n:.2f} % (Wilson 95 % CI {100*lo:.2f} to {100*hi:.2f} %)",
             f"offers where at least one model says critical: {any_crit} ({100*any_crit/n:.2f} %)", f"unanimous verdicts: {unanimous} ({100*unanimous/n:.2f} %)",
             f"Fleiss' kappa over the three verdict classes: {fleiss_kappa(counters):.3f}", "", "## Per-model verdict distribution"]
    for m in models:
        c = Counter(verdict[m][i] for i in ids); lines.append(f"- {m}: " + ", ".join(f"{v} {c.get(v,0)}" for v in VERDICTS))
    lines += ["", f"flagged for manual review: {len(flagged)} (flagged_for_review.csv)", f"calibration sample: {min(a.sample, len(unanimous_ok))} unanimous error-free offers (calibration_sample.csv)"]
    open(f"{OUT}/audit_summary.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
