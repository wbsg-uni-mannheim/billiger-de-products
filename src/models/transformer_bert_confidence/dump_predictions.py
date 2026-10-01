"""Write per-pair predictions for an already-trained RoBERTa baseline checkpoint.

The training script only saves aggregate metrics, so runs finished before that was
fixed have no per-pair output and cannot feed the per-category analysis. This script
reloads a finished run and writes the missing prediction files. It never trains.

    python src/models/transformer_bert_confidence/dump_predictions.py \
        --run_dir results/generated/roberta_fix3/de/products50cc50rnd000un-small/0 \
        --test_file data/processed/gold-standards_adjusted/preprocessed_products50cc50rnd000un_gs.pkl.gz

Output, next to the run: baseline_predictions.csv, baseline_predictions_un050.csv,
baseline_predictions_un100.csv, each with pair_id, label, probability, prediction --
the same format the analysis code already consumes.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import BaselineClassificationDataset  # noqa: E402


def pair_ids_for(source_file, dataset):
    """`_prepare_data` keeps only features/label, so pair_id is read back from the
    source file. Row order and labels are asserted identical before it is used."""
    src = pd.read_pickle(source_file, compression="gzip")
    assert len(src) == len(dataset.data), (len(src), len(dataset.data))
    assert (np.asarray(src["label"]) == np.asarray(dataset.data["label"])).all()
    return src["pair_id"].astype(str).tolist()


def dump(model, dataset, collator, out_path, batch_size, device, source_file):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits.append(model(**batch).logits.float().cpu())
    logits = torch.cat(logits)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()

    pd.DataFrame({
        "pair_id": pair_ids_for(source_file, dataset),
        "label": np.asarray(dataset.data["label"]).astype(int),
        "probability": probs,
        "prediction": preds,
    }).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(preds)} rows)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="directory holding model.safetensors")
    p.add_argument("--test_file", required=True, help="the 000un gold standard; 050un/100un are derived")
    p.add_argument("--tokenizer", default="roberta-base")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} run_dir={args.run_dir}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.run_dir, num_labels=2)
    model.to(device).eval()

    targets = [(args.test_file, "baseline_predictions.csv")]
    if "000un" in args.test_file:
        targets += [(args.test_file.replace("000un", "050un"), "baseline_predictions_un050.csv"),
                    (args.test_file.replace("000un", "100un"), "baseline_predictions_un100.csv")]

    collator = None
    for test_file, name in targets:
        ds = BaselineClassificationDataset(test_file, dataset_type="test", tokenizer=args.tokenizer,
                                           dataset="lspc")
        if collator is None:
            collator = DataCollatorWithPadding(tokenizer=ds.tokenizer, padding="longest", max_length=256)
        dump(model, ds, collator, os.path.join(out_dir, name), args.batch_size, device, test_file)


if __name__ == "__main__":
    main()
