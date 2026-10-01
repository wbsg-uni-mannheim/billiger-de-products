"""Compare a filled-in audit_labels.csv against the gold labels of the sample."""
import csv, sys, math, os
OUT = os.path.dirname(os.path.abspath(__file__))
mine = {r["pair_id"]: r["my_label"].strip() for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/audit_labels.csv"))}
key = {r["pair_id"]: r for r in csv.DictReader(open(f"{OUT}/sample_key.csv"))}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

for cls, name in ((None, "all"), ("1", "gold positives"), ("0", "gold negatives")):
    n = d = u = 0
    for pid, g in key.items():
        if cls and g["gold_label"] != cls: continue
        v = mine.get(pid, "")
        if v == "": continue
        if v == "unsure": u += 1; continue
        n += 1; d += (v != g["gold_label"])
    lo, hi = wilson(d, n)
    print(f"{name:>15}: {d}/{n} disagree = {100*d/n if n else 0:.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]  ({u} unsure)")

print("\ndisagreements:")
for pid, g in key.items():
    v = mine.get(pid, "")
    if v not in ("0", "1") or v == g["gold_label"]: continue
    print(f"  #{g['n']:>3} {pid}  gold={g['gold_label']} yours={v}  hard_neg={g['is_hard_negative']}  {g['configs']}")
