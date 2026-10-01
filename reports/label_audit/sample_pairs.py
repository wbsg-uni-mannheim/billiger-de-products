"""Draw a stratified 50 pos / 50 neg sample from the German test sets for a blind re-label."""
import gzip, json, glob, random, csv, os, collections

SEED = 20260820
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

de_files = sorted(glob.glob(f"{ROOT}/data/solute_de/gold-standards_adjusted/*.json.gz"))
en_files = sorted(glob.glob(f"{ROOT}/data/solute_en/gold-standards_adjusted/*.json.gz"))

pairs, configs = {}, collections.defaultdict(list)
for p in de_files:
    cfg = os.path.basename(p).replace("products", "").replace("_gs.json.gz", "")
    with gzip.open(p, "rt") as f:
        for line in f:
            d = json.loads(line)
            configs[d["pair_id"]].append(cfg)
            pairs.setdefault(d["pair_id"], d)

en = {}
for p in en_files:
    with gzip.open(p, "rt") as f:
        for line in f:
            d = json.loads(line)
            en.setdefault(d["pair_id"], d)

pos = sorted(k for k, d in pairs.items() if d["label"] == 1)
neg = sorted(k for k, d in pairs.items() if d["label"] == 0)
rng = random.Random(SEED)
sample = rng.sample(pos, 50) + rng.sample(neg, 50)
rng.shuffle(sample)

items, key = [], []
for i, pid in enumerate(sample, 1):
    d = pairs[pid]
    e = en.get(pid, {})
    items.append({
        "n": i, "pair_id": pid,
        "left": {"id": d["id_left"], "brand": d["brand_left"], "name": d["name_left"],
                 "price": d["price_left"], "desc": d["desc_left"], "name_en": e.get("name_left", "")},
        "right": {"id": d["id_right"], "brand": d["brand_right"], "name": d["name_right"],
                  "price": d["price_right"], "desc": d["desc_right"], "name_en": e.get("name_right", "")},
    })
    key.append({"n": i, "pair_id": pid, "gold_label": d["label"],
                "is_hard_negative": d.get("is_hard_negative"),
                "product_id_left": d["product_id_left"], "product_id_right": d["product_id_right"],
                "configs": ";".join(sorted(set(configs[pid])))})

with open(f"{OUT}/sample_blind.json", "w") as f:
    json.dump(items, f, ensure_ascii=False)
with open(f"{OUT}/sample_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key[0]))
    w.writeheader(); w.writerows(key)

# flat CSV fallback for labelling in a spreadsheet (no gold label)
with open(f"{OUT}/sample_blind.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "pair_id", "brand_left", "name_left", "price_left",
                "brand_right", "name_right", "price_right", "my_label", "note"])
    for it in items:
        w.writerow([it["n"], it["pair_id"], it["left"]["brand"], it["left"]["name"], it["left"]["price"],
                    it["right"]["brand"], it["right"]["name"], it["right"]["price"], "", ""])

print("sampled", len(items), "pairs; positives in key:", sum(k["gold_label"] for k in key))
print("pool: %d unique pairs, %d positives" % (len(pairs), len(pos)))
