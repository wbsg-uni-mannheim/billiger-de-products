import json
import os
import datetime

def save_blocking_pairs(pairs, dataset, k, split, out_dir):
    """
    pairs: list of dicts from query_table.materialize_pairs()
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"blocking_pairs_{dataset}_k{k}_{split}.jsonl")

    with open(out_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "ltableid": pair["ltableid"],
                "rtableid": pair["rtableid"],
                "match": pair.get("match", None),
                "dataset": pair.get("dataset", dataset),
                "matched_table": pair.get("matched_table", None),
            }
            f.write(json.dumps(record) + "\n")
