#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WDC Block Benchmark Generator
--------------------------------
Implements the full 4-step benchmark generation process:
1. Select a difficult variant of the WDC Products benchmark as seed dataset.
2. Split into datasets A and B and deduplicate.
3. Enlarge each with additional non-matching offers from WDC Product Data Corpus V2020.
4. Generate three development set sizes (~1k, ~5k, ~20k pairs) with 50% and 100% unseen variants.
"""

import os
import random
import csv

import numpy as np
import pandas as pd

from gensim.models import FastText


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

np.random.seed(42)
random.seed(42)

DATA_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name.pkl.gz'
SEED_TRAIN = 'data/derived/training-sets/products80cc20rnd000un_train_large.json.gz'
SEED_TEST = 'data/derived/gold-standards_adjusted/products80cc20rnd050un_gs.json.gz'
SEED_VALID = 'data/derived/validation-sets/products80cc20rnd000un_valid_small.json.gz'
OUTPUT_DIR = 'data/blocking_short_desc'
MAX_DESC_CHARS = 256  

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def load_corpus():
    """
    Load the WDC Product Data Corpus V2020 (or compatible) and keep only
    the relevant columns.
    """
    corpus = pd.read_pickle(DATA_PATH, compression='gzip')
    corpus = corpus[['id', 'product_id', 'name', 'desc', 'brand', 'price', 'shop_cat']]
    corpus = corpus.dropna(subset=['name'])

    # clean entire corpus
    corpus = clean_text(corpus)
    corpus = truncate_description(corpus, MAX_DESC_CHARS)

    return corpus

def clean_text(df):
    df = df.copy()  # ensure full copy, removes chained-assignment warnings
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "string":
            df.loc[:, col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)
                .str.replace('"', "'", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"\\+", " ", regex=True)
                .str.strip()
            )
    return df

def truncate_description(df, max_chars=None):
    """
    Optionally truncate the 'desc' column to max_chars characters.
    If max_chars is None, do nothing.
    """
    if max_chars is None:
        return df

    df = df.copy()
    if 'desc' in df.columns:
        df['desc'] = (
            df['desc']
            .astype(str)
            .str.slice(0, max_chars)
        )
    return df

# --------------------------------------------------------------------
# Step 1 – Load seed dataset (train/valid/test)
# --------------------------------------------------------------------

def load_seed_splits():
    """
    Load original EM benchmark splits that act as seed for WDC-Block.
    """
    train = pd.read_json(SEED_TRAIN, compression='gzip', lines=True)
    valid = pd.read_json(SEED_VALID, compression='gzip', lines=True)
    test = pd.read_json(SEED_TEST, compression='gzip', lines=True)

    print(f"Loaded seed splits: train={len(train)}, valid={len(valid)}, test={len(test)}")
    return clean_text(train), clean_text(valid), clean_text(test)


# --------------------------------------------------------------------
# Step 2 – Split into A and B (by id)
# --------------------------------------------------------------------

def split_into_a_b(train_df, valid_df, test_df):
    """
    Split into datasets A and B by id sets using ALL seed splits.
    """
    
    a_ids = set(train_df['id_left'].unique()) \
            | set(valid_df['id_left'].unique()) \
            | set(test_df['id_left'].unique())

    b_ids = set(train_df['id_right'].unique()) \
            | set(valid_df['id_right'].unique()) \
            | set(test_df['id_right'].unique())

    print(f"Initial A ids: {len(a_ids)}, B ids: {len(b_ids)}")
    return a_ids, b_ids


# --------------------------------------------------------------------
# Step 3 – Enlarge datasets and build tableA/tableB for each size
# --------------------------------------------------------------------

def save_set(df, path):
    df['id'] = df['id'].astype(int)
    df['product_id'] = df['product_id'].astype(int)

    df = clean_text(df)
    #quoting=csv.QUOTE_MINIMAL,
    #quotechar='"',
    #escapechar='\\',
    df.to_csv(
        path,
        index=False, #
        encoding='utf-8'
    )
    print(f"Saved {path}")


def resize_ids(corpus, base_ids, required_ids, target_size, side_name):
    """
    Create the final set of ids for table A or B given:
    - base_ids: ids coming from seed EM pairs (id_left/right)
    - required_ids: ids that MUST be present because they appear in train/valid/test
    - target_size: desired size (5k, 200k, 2M, ...)
    """
    corpus_ids = set(corpus['id'].unique())

    # intersect with corpus (in case some ids are missing)
    base_ids = set(base_ids) & corpus_ids
    required_ids = set(required_ids) & corpus_ids

    ids = base_ids | required_ids

    if len(required_ids) > target_size:
        print(f"[WARN] Required {side_name} IDs ({len(required_ids)}) > target_size ({target_size}). "
              f"Using all required IDs; effective size will be {len(ids)}.")
        return list(ids)

    # If still smaller than target_size, add extra non-matching ids
    if len(ids) < target_size:
        remaining = target_size - len(ids)
        nonmatching = list(corpus_ids - ids)
        if remaining > 0 and nonmatching:
            extra = random.sample(nonmatching, min(remaining, len(nonmatching)))
            ids.update(extra)

    # If bigger than target_size, drop only IDs that are not required
    if len(ids) > target_size:
        optional = list(ids - required_ids)
        excess = len(ids) - target_size
        if excess > 0 and optional:
            to_remove = set(random.sample(optional, min(excess, len(optional))))
            ids = ids - to_remove

    return list(ids)


def build_tables_for_size(
    corpus,
    size_name,
    base_a_ids,
    base_b_ids,
    required_a,
    required_b,
    target_sizes
):
    """
    Build tableA/tableB for a specific size (small/medium/large) and
    ensure that ALL ids used in train_s/valid_s/test_s are contained.
    """
    size_a, size_b = target_sizes[size_name]
    size_dir = os.path.join(OUTPUT_DIR, size_name)
    os.makedirs(size_dir, exist_ok=True)

    final_a_ids = resize_ids(corpus, base_a_ids, required_a, size_a, side_name=f"A/{size_name}")
    final_b_ids = resize_ids(corpus, base_b_ids, required_b, size_b, side_name=f"B/{size_name}")

    table_a = corpus[corpus['id'].isin(final_a_ids)].copy()
    table_b = corpus[corpus['id'].isin(final_b_ids)].copy()
    
    table_a['id_placeholder'] = table_a['id']
    table_b['id_placeholder'] = table_b['id']

    # Reset IDs to 0..N-1 after final selection 
    table_a = table_a.reset_index(drop=True) 
    table_b = table_b.reset_index(drop=True) 
    table_a['id'] = table_a.index 
    table_b['id'] = table_b.index

    cols = ['id', 'brand', 'name', 'desc', 'price', 'product_id', 'shop_cat', 'id_placeholder']
    table_a = table_a[cols].set_index('id', drop=False)
    table_b = table_b[cols].set_index('id', drop=False)

    table_a = clean_text(table_a)
    table_b = clean_text(table_b)

    return table_a, table_b


# --------------------------------------------------------------------
# Schritt 4 – Development-Sets erzeugen (train/valid/test CSVs)
# --------------------------------------------------------------------

def convert_to_wdc_blocking_format(df, table_a, table_b, size_name):
    """
    Map id_left/right to tableA.id / tableB.id (ltable_id / rtable_id).
    """
    id_map_a = table_a.set_index('id_placeholder')['id'].to_dict() 
    id_map_b = table_b.set_index('id_placeholder')['id'].to_dict() 
    df['ltable_id'] = df['id_left'].map(id_map_a) 
    df['rtable_id'] = df['id_right'].map(id_map_b)



    # Optional: sanity check
    missing_l = df['ltable_id'].isna().sum()
    missing_r = df['rtable_id'].isna().sum()
    if missing_l > 0 or missing_r > 0:
        print(f"[WARN] Missing ltable_id: {missing_l}, missing rtable_id: {missing_r}")

    size_dir = os.path.join(OUTPUT_DIR, size_name)
    os.makedirs(size_dir, exist_ok=True)
    table_a = table_a.drop(columns=['id_placeholder'])
    table_b = table_b.drop(columns=['id_placeholder'])
    if (not os.path.exists(os.path.join(size_dir, "tableA.csv")) and not os.path.exists(os.path.join(size_dir, "tableB.csv"))):
        save_set(table_a, os.path.join(size_dir, "tableA.csv"))
        save_set(table_b, os.path.join(size_dir, "tableB.csv"))
    
    df['ltable_id'] = df['ltable_id'].astype(int)
    df['rtable_id'] = df['rtable_id'].astype(int)
    df['label'] = df['label'].astype(int)
    df['product_id_left'] = df['product_id_left'].astype(int)
    df['product_id_right'] = df['product_id_right'].astype(int)

    return df[[
        "ltable_id",
        "rtable_id",
        "label",
        "product_id_left",
        "product_id_right",
        "pair_id"
    ]]


def stratified_downsample(df, target_n, by_cols=["label"], random_state=42): 
    """ Downsample df to target_n rows while approximately preserving label distribution. """ 
    n = min(target_n, len(df)) 
    if n == len(df): 
        return df.sample(frac=1.0, random_state=random_state) 
    grp = df.groupby(by_cols) 
    counts = grp.size().reset_index(name="count") 
    counts["prop"] = counts["count"] / counts["count"].sum() 
    counts["target"] = (counts["prop"] * n).round().astype(int) 
    diff = n - counts["target"].sum() 
    if diff != 0: 
        counts = counts.sort_values("prop", ascending=False) 
        counts.iloc[:abs(diff), counts.columns.get_loc("target")] += int(np.sign(diff)) 
        counts = counts.sort_index() 
    parts = [] 
    for _, row in counts.iterrows(): 
        if len(by_cols) == 1: 
            subset = grp.get_group(row[by_cols[0]]) 
        else: 
            subset = grp.get_group(tuple(row[c] for c in by_cols)) 
        take = int(min(row["target"], len(subset))) 
        if take > 0: 
            parts.append(subset.sample(n=take, random_state=random_state)) 
        out = pd.concat(parts, ignore_index=True) if parts else df.head(0) 
    return out.sample(frac=1.0, random_state=random_state)


def save_split(df, out_path):
    df = clean_text(df)
    #quoting=csv.QUOTE_MINIMAL,
    #quotechar='"',
    #escapechar='\\',
    df.to_csv(
        out_path,
        index=False,
        encoding='utf-8'
    )
    print(f"Saved {out_path}")


def prepare_block_devsets(corpus, base_a_ids, base_b_ids, train_seed, valid_seed, test_seed):


    print("Original sizes:",
          len(train_seed), "train,",
          len(valid_seed), "valid,",
          len(test_seed), "test")

    # Sizes from WDC-Block (Table 2 style)
    sizes = {
        "small":  {"train": 800,   "valid": 200},
        "medium": {"train": 4000,  "valid": 1000},
        "large":  {"train": 16000, "valid": 4000}
    }

    target_table_sizes = {
        "small":  (5_000, 5_000),
        "medium": (5_000, 200_000),
        "large":  (100_000, 2_000_000),
    }

    for size_name, sz in sizes.items():
        print(f"\n=== Building {size_name} benchmark ===")

        size_dir = os.path.join(OUTPUT_DIR, size_name)
        os.makedirs(size_dir, exist_ok=True)

        # 1) Downsample splits for this size
        train_s = stratified_downsample(train_seed, sz["train"], by_cols=["label"])
        valid_s = stratified_downsample(valid_seed, sz["valid"], by_cols=["label"])
        test_s = test_seed.copy()  # test always unchanged

        print(f"{size_name}: train={len(train_s)}, valid={len(valid_s)}, test={len(test_s)}")

        required_a = set(train_s['id_left']) \
                 | set(valid_s['id_left']) \
                 | set(test_s['id_left'])

        required_b = set(train_s['id_right']) \
                 | set(valid_s['id_right']) \
                 | set(test_s['id_right'])
        # 2) Build tableA/tableB for this size, ensuring all ids from splits exist
        table_a, table_b = build_tables_for_size(
            corpus,
            size_name,
            base_a_ids,
            base_b_ids,
            required_a,
            required_b,
            target_table_sizes
        )

        # 3) Map ids to ltable_id/rtable_id
        train_m = convert_to_wdc_blocking_format(train_s, table_a, table_b, size_name)
        valid_m = convert_to_wdc_blocking_format(valid_s, table_a, table_b, size_name)
        test_m  = convert_to_wdc_blocking_format(test_s, table_a, table_b, size_name)

        # 4) Save splits
        save_split(train_m, os.path.join(size_dir, "train.csv"))
        save_split(valid_m, os.path.join(size_dir, "valid.csv"))
        save_split(test_m,  os.path.join(size_dir, "test.csv"))


# --------------------------------------------------------------------
# Core benchmark assembly
# --------------------------------------------------------------------

def generate_block_benchmark():
    corpus = load_corpus()
    train_seed, valid_seed, test_seed = load_seed_splits()

    # derive base id sets for A and B from all seed splits
    base_a_ids, base_b_ids = split_into_a_b(train_seed, valid_seed, test_seed)

    # build dev sets + tableA/tableB for each size
    prepare_block_devsets(corpus, base_a_ids, base_b_ids, train_seed, valid_seed, test_seed)


if __name__ == '__main__':
    generate_block_benchmark()
