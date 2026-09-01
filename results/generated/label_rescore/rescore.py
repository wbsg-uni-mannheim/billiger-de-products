"""Rescore HierGAT and Ditto per-pair test predictions against the released
(audited) gold standards. Read-only on predictions and gold standards."""
import pandas as pd, numpy as np, glob, re, json, os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support as prf

T = "/ceph/aasteine/kelagin_backup/work/kelagin/Entity-Matching-Pipeline-for-German-Product-Data---Master-Thesis"
B = "/ceph/aasteine/kelagin_backup/billiger-de-products"
OUT = f"{B}/results/generated/label_rescore"
os.makedirs(OUT, exist_ok=True)

PRED_DIRS = {("hiergat","de"): f"{T}/src/models/hiergat/output/prediction_de",
             ("hiergat","en"): f"{T}/src/models/hiergat/output_en/prediction",
             ("ditto","de"):   f"{T}/src/models/ditto/output/prediction_de",
             ("ditto","en"):   f"{T}/src/models/ditto/output_en/prediction"}
PRE = {"de": f"{T}/data/processed/gold-standards_adjusted",
       "en": f"{T}/data/processed_en/gold-standards_adjusted"}
REL = {"de": f"{B}/data/processed/gold-standards_adjusted",
       "en": f"{B}/data/processed_en/gold-standards_adjusted"}
CCMAP = {"20":"20cc80rnd","50":"50cc50rnd","80":"80cc20rnd"}
SIZES = ["small","medium","large"]; UNS = ["000un","050un","100un"]

def parse(stem):
    m = re.search(r"final_(small|medium|large)_(\d+)cc\d+rnd\d+un_.*?_id=(\d+)_.*?_predictions(?:_(\d+))?$", stem)
    if not m: return None
    size, cc, seed, testun = m.groups()
    return size, CCMAP[cc], (testun + "un") if testun else "000un", int(seed)

_gc = {}
def gold(kind, lang, cc, un):
    k = (kind, lang, cc, un)
    if k not in _gc:
        _gc[k] = pd.read_pickle(f"{(PRE if kind=='pre' else REL)[lang]}/preprocessed_products{cc}{un}_gs.pkl.gz")
    return _gc[k]

# ---------------------------------------------------------------- index
files = []
for (model, lang), d in PRED_DIRS.items():
    for f in sorted(glob.glob(d + "/*.csv")):
        p = parse(Path(f).stem)
        if p is None:
            raise SystemExit(f"unparsed prediction file: {f}")
        size, cc, un, seed = p
        files.append(dict(model=model, language=lang, cc=cc, size=size, un=un, seed=seed, path=f))
idx = pd.DataFrame(files)

missing = []
for model in ["hiergat","ditto"]:
    for lang in ["de","en"]:
        have = set(zip(*[idx[(idx.model==model)&(idx.language==lang)][c] for c in ["cc","size","un","seed"]]))
        for cc in CCMAP.values():
            for size in SIZES:
                for un in UNS:
                    for seed in range(3):
                        if (cc,size,un,seed) not in have:
                            missing.append(dict(model=model, language=lang, cc=cc, size=size, un=un, seed=seed))
pd.DataFrame(missing).to_csv(f"{OUT}/missing_prediction_files.csv", index=False)

# ---------------------------------------------------------------- scoring
per_seed, preaudit, joins = [], [], []
DITTO_RULE_OK = True
for r in idx.itertuples():
    pred = pd.read_csv(r.path)
    gpre = gold("pre", r.language, r.cc, r.un)
    grel = gold("rel", r.language, r.cc, r.un)

    if len(pred) != len(gpre):
        raise SystemExit(f"row-count mismatch {r.path}: {len(pred)} vs pre-audit {len(gpre)}")
    if "threshold" in pred.columns:
        rule = (pred.probability > pred.threshold).astype(int)
        if not (rule.to_numpy() == pred.prediction.to_numpy()).all():
            DITTO_RULE_OK = False

    # positional alignment: prediction row i == pre-audit gold row i
    p = pd.DataFrame({"pair_id": gpre.pair_id.to_numpy(),
                      "prediction": pred.prediction.to_numpy(),
                      "label_embedded": pred.label.to_numpy(),
                      "label_preaudit": gpre.label.to_numpy()})

    j = p.merge(grel[["pair_id","label"]].rename(columns={"label":"label_released"}),
                on="pair_id", how="outer", indicator=True)
    n_pred_only = int((j._merge == "left_only").sum())     # dropped by the audit
    n_gold_only = int((j._merge == "right_only").sum())    # released row with no prediction
    m = j[j._merge == "both"]
    joins.append(dict(model=r.model, language=r.language, cc=r.cc, size=r.size, un=r.un, seed=r.seed,
                      n_prediction_rows=len(pred), n_gold_rows=len(grel), n_matched=len(m),
                      n_unmatched_prediction_only=n_pred_only, n_unmatched_gold_only=n_gold_only))

    pr, rc, f1, _ = prf(m.label_released, m.prediction, average="binary", zero_division=0)
    per_seed.append(dict(model=r.model, language=r.language, cc=r.cc, size=r.size, un=r.un, seed=r.seed,
                         precision=pr*100, recall=rc*100, f1=f1*100,
                         n_pairs=len(m), n_unmatched=n_pred_only + n_gold_only))

    e = prf(p.label_embedded, p.prediction, average="binary", zero_division=0)
    a = prf(p.label_preaudit, p.prediction, average="binary", zero_division=0)
    preaudit.append(dict(model=r.model, language=r.language, cc=r.cc, size=r.size, un=r.un, seed=r.seed,
                         f1_embedded_label=e[2]*100, precision_embedded_label=e[0]*100, recall_embedded_label=e[1]*100,
                         f1_preaudit_gold=a[2]*100,
                         n_rows=len(p), n_pos_embedded=int(p.label_embedded.sum()),
                         n_pos_preaudit=int(p.label_preaudit.sum()),
                         embedded_equals_preaudit=bool((p.label_embedded.to_numpy()==p.label_preaudit.to_numpy()).all())))

ps = pd.DataFrame(per_seed).sort_values(["model","language","cc","size","un","seed"])
ps.to_csv(f"{OUT}/metrics_per_seed.csv", index=False)
mean = (ps.groupby(["model","language","cc","size","un"])
          .agg(f1_mean=("f1","mean"), f1_std=("f1","std"),
               precision_mean=("precision","mean"), recall_mean=("recall","mean"),
               n_seeds=("seed","count"), n_pairs=("n_pairs","max"),
               n_unmatched=("n_unmatched","max")).reset_index())
mean.to_csv(f"{OUT}/metrics_mean.csv", index=False)
pd.DataFrame(preaudit).to_csv(f"{OUT}/validation_preaudit_per_seed.csv", index=False)
pd.DataFrame(joins).to_csv(f"{OUT}/join_report.csv", index=False)
va = (pd.DataFrame(preaudit).groupby(["model","language","cc","size","un"])
        .agg(f1_embedded_mean=("f1_embedded_label","mean"),
             f1_preaudit_gold_mean=("f1_preaudit_gold","mean"),
             n_seeds=("seed","count")).reset_index())
va.to_csv(f"{OUT}/validation_preaudit_mean.csv", index=False)
print("ditto decision rule (probability > threshold) reproduces stored prediction column:", DITTO_RULE_OK)
print("files scored:", len(ps), " missing:", len(missing))
print("wrote", OUT)
