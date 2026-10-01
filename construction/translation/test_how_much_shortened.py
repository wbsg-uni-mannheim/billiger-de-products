import tiktoken
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

# Für GPT-4/5 Modelle (cl100k_base Encoding)
enc = tiktoken.get_encoding("cl100k_base")

def count_gpt_tokens(text):
    if not text:
        return 0
    return len(enc.encode(text))

def safe_str(x):
    if x is None:
        return ""
    return str(x)

def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def collect_product_stats_from_dataset(path):
    """
    Returns dict: product_id -> stats
    Stats tracked as MAX observed over occurrences:
      - max_desc_chars
      - max_desc_tokens
      - max_name_chars
      - max_name_tokens
      - any_desc_endswith_ellipsis
    """
    stats = {}

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for side in ("left", "right"):
                pid = str(r.get(f"id_{side}", ""))
                if not pid:
                    continue

                name = safe_str(r.get(f"name_{side}", ""))
                desc = safe_str(r.get(f"desc_{side}", ""))

                desc_chars = len(desc)
                desc_tokens = count_gpt_tokens(desc)
                name_chars = len(name)
                name_tokens = count_gpt_tokens(name)

                d = stats.get(pid)
                if d is None:
                    d = {
                        "id": pid,
                        "name_chars": name_chars,
                        "name_tokens": name_tokens,
                        "desc_chars": desc_chars,
                        "desc_tokens": desc_tokens,
                    }
                    
                else:
                    # IMPORTANT: take MAX over occurrences
                    stats[pid]["desc_chars"] = max(stats[pid]["desc_chars"], desc_chars)
                    stats[pid]["desc_tokens"] = max(stats[pid]["desc_tokens"], desc_tokens)
                    stats[pid]["name_chars"] = max(stats[pid]["name_chars"], name_chars)
                    stats[pid]["name_tokens"] = max(stats[pid]["name_tokens"], name_tokens)

                stats[pid] = d
    return stats

def name_desc_shortened(df):
    # create for every id a boolean column indicating if the description was shortened in the English translation
    # if the english description or name is more than 20% shorter than the German description make desc_shortened True, else False
    df = df.fillna(0)
    

    df["desc_shortened_30p"] = (
        (df["desc_tokens_en"] < 0.7 * df["desc_tokens_de"])
    )

    df["name_shortened_30p"] = (
        (df["name_tokens_en"] < 0.7 * df["name_tokens_de"])
    )
    df["desc_shortened_40p"] = (
        (df["desc_tokens_en"] < 0.6 * df["desc_tokens_de"])
    )

    df["name_shortened_40p"] = (
        (df["name_tokens_en"] < 0.6 * df["name_tokens_de"])
    )

    df["desc_shortened_50p"] = (
        (df["desc_tokens_en"] < 0.5 * df["desc_tokens_de"])
    )

    df["name_shortened_50p"] = (
        (df["name_tokens_en"] < 0.5 * df["name_tokens_de"])
    )
    return df

    

def main():
    folders = ["validation-sets"]#["training-sets", "validation-sets", "gold-standards_adjusted"]
    de_folder = "data/derived"
    en_folder = "data/derived_en_new"

    for folder in folders:
        for file in os.listdir(os.path.join(en_folder, folder)):
            if file.endswith(".json.gz") and not file.__contains__("multi"):
                de_path = os.path.join(de_folder, folder, file)
                en_path = os.path.join(en_folder, folder, file)

                print(f"Processing {file}...")
                de_stats = collect_product_stats_from_dataset(Path(de_path))
                en_stats = collect_product_stats_from_dataset(Path(en_path))

                df_de = pd.DataFrame(de_stats.values())
                df_en = pd.DataFrame(en_stats.values())               
                df = pd.merge(
                        df_de,
                        df_en,
                        on="id",
                        suffixes=("_de", "_en"),
                        how="outer"
                    )
                df = name_desc_shortened(df)

                out_csv = f"src/translate_to_english/analysis/translation_impact_{folder}_{file.replace('.json.gz', '.csv')}"
                df.to_csv(out_csv, index=False)
                print(f"Written per-product stats to: {out_csv}")
                #check how many products have their description shortened
                num_shortened_30p = df["desc_shortened_30p"].sum()
                num_name_shortened_30p = df["name_shortened_30p"].sum()
                num_shortened_40p = df["desc_shortened_40p"].sum()
                num_name_shortened_40p = df["name_shortened_40p"].sum()
                num_shortened_50p = df["desc_shortened_50p"].sum()
                num_name_shortened_50p = df["name_shortened_50p"].sum()
                print(f"Analysis for {folder} {file}:")
                print(f"Number of products with 30% shortened description: {num_shortened_30p} / {len(df)}")
                print(f"Number of products with 30% shortened name: {num_name_shortened_30p} / {len(df)}")
                print(f"Number of products with 40% shortened description: {num_shortened_40p} / {len(df)}")
                print(f"Number of products with 40% shortened name: {num_name_shortened_40p} / {len(df)}")
                print(f"Number of products with 50% shortened description: {num_shortened_50p} / {len(df)}")
                print(f"Number of products with 50% shortened name: {num_name_shortened_50p} / {len(df)}")
                print("")

ANALYSIS_DIR = "src/translate_to_english/analysis"
OUTPUT_EXCEL = os.path.join(ANALYSIS_DIR, "translation_impact_MASTER_SUMMARY.xlsx")

def map_split(folder_name: str):
    if "training-sets" in folder_name:
        return "train"
    elif "validation-sets" in folder_name:
        return "validation"
    elif "gold-standards" in folder_name:
        return "test"
    else:
        return "unknown"

def extract_cc(filename: str):
    match = re.search(r"(\d{2})cc", filename)
    return int(match.group(1)) if match else None

def extract_unseen(filename: str):
    match = re.search(r"(\d{3})un", filename)
    return int(match.group(1)) if match else None

def extract_size(filename: str):
    if filename.__contains__("small"):
        return "small"
    elif filename.__contains__("medium"):
        return "medium"
    elif filename.__contains__("large"):
        return "large"
    else:
        return ""

def create_Excel():
    ANALYSIS_DIR = "src/translate_to_english/analysis"
    OUTPUT_EXCEL = os.path.join(ANALYSIS_DIR, "translation_impact_MASTER_SUMMARY.xlsx")
    rows = []

    for file in os.listdir(ANALYSIS_DIR):
        if not file.endswith(".csv"):
            continue
        if "translation_impact_" not in file:
            continue

        file_path = os.path.join(ANALYSIS_DIR, file)

        # Load per-product stats
        df = pd.read_csv(file_path)

        total = len(df)

        num_shortened_30p = df["desc_shortened_30p"].sum()
        num_name_shortened_30p = df["name_shortened_30p"].sum()
        num_shortened_40p = df["desc_shortened_40p"].sum()
        num_name_shortened_40p = df["name_shortened_40p"].sum()
        num_shortened_50p = df["desc_shortened_50p"].sum()
        num_name_shortened_50p = df["name_shortened_50p"].sum()

        split = map_split(file)

        cc_value = extract_cc(file)
        unseen_value = extract_unseen(file)
        size = extract_size(file)

        rows.append({
            "split": split,
            "cc_percent": cc_value,
            "unseen_percent": unseen_value,
            "size": size,
            "total_products": total,
            "pct_30_shorter_desc": num_shortened_30p,
            "pct_30_shorter_name": num_name_shortened_30p,
            "pct_40_shorter_desc": num_shortened_40p,
            "pct_40_shorter_name": num_name_shortened_40p,
            "pct_50_shorter_desc": num_shortened_50p,
            "pct_50_shorter_name": num_name_shortened_50p,
        })

    summary_df = pd.DataFrame(rows)

    # Sort nicely
    summary_df = summary_df.sort_values(
        by=["split", "cc_percent", "unseen_percent"]
    ).reset_index(drop=True)

    summary_df.to_excel(OUTPUT_EXCEL, index=False)

    print("Master summary written to:")
    print(OUTPUT_EXCEL)



if __name__ == "__main__":
    main()
    create_Excel()
