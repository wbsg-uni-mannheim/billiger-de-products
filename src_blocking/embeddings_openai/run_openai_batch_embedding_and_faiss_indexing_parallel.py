#!/usr/bin/env python3

import argparse
import json
import time
import os
import numpy as np
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)

client = OpenAI(api_key=API_key)

MAX_WORKERS = 5
SPLIT_THRESHOLD = 6250


# --------------------------------------------------
# Serialization
# --------------------------------------------------

def serialize_record(row):

    parts = []

    if pd.notna(row.get("brand")):
        parts.append(f"[BRAND] {row['brand']}")

    if pd.notna(row.get("name")):
        parts.append(f"[NAME] {row['name']}")

    if pd.notna(row.get("desc")):
        parts.append(f"[DESC] {row['desc']}")

    if pd.notna(row.get("price")):
        parts.append(f"[PRICE] {row['price']}")

    return " ".join(parts)


def load_and_serialize(csv_path):

    df = pd.read_csv(csv_path)
    return df.apply(serialize_record, axis=1).tolist()



def clean_text(text):
    if not isinstance(text, str):
        print("Warning: Non-string value encountered during text cleaning. Returning empty string.")
        return ""

    text = text.encode("utf-8", "ignore").decode("utf-8")

    # remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip()
    text = text[:8000]

    return text

# --------------------------------------------------
# Batch File Creation
# --------------------------------------------------

def write_batch_files(texts, out_dir, chunk_size=50000):

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(0, len(texts), chunk_size):

        path = os.path.join(out_dir, f"batch_{i//chunk_size:04d}.jsonl")

        if os.path.exists(path):
            paths.append(path)
            continue

        with open(path, "w", encoding="utf-8") as f:

            for j, text in enumerate(texts[i:i+chunk_size]):
                text = clean_text(text)

                f.write(json.dumps({
                    "custom_id": str(i + j),
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": text
                    }
                }) + "\n")

        paths.append(path)

    return paths


# --------------------------------------------------
# OpenAI Batch Helpers
# --------------------------------------------------

def submit_batch(jsonl_path):

    file_obj = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/embeddings",
        completion_window="24h"
    )

    print("Submitted:", batch.id)

    return batch.id


def wait_for_batch(batch_id):

    while True:

        batch = client.batches.retrieve(batch_id)

        print(batch_id, batch.status)
        

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch

        time.sleep(60)


# --------------------------------------------------
# Download embeddings
# --------------------------------------------------

def download_embeddings(batch, emb_dir, prefix, batch_idx):

    # retry loop for network download
    for attempt in range(5):

        try:
            print(f"Downloading batch result for {prefix} chunk {batch_idx}")
            response = client.files.content(batch.output_file_id)
            raw_bytes = response.read()
            break

        except Exception as e:
            print(f"Download failed (attempt {attempt+1}/5)")
            print(e)
            time.sleep(10)

    else:
        raise RuntimeError("Failed to download batch result after retries")

    
    lines = raw_bytes.decode("utf-8").splitlines()

    embeddings = []
    ids = []

    for line in lines:

        obj = json.loads(line)

        idx = int(obj["custom_id"])
        emb = obj["response"]["body"]["data"][0]["embedding"]

        embeddings.append(emb)
        ids.append(idx)

    X = np.array(embeddings, dtype=np.float32)
    ids = np.array(ids, dtype=np.int32)

    emb_path = os.path.join(emb_dir, f"chunk_{prefix}_{batch_idx}.npy")
    id_path = os.path.join(emb_dir, f"chunk_ids_{prefix}_{batch_idx}.npy")

    np.save(emb_path, X)
    np.save(id_path, ids)

    os.makedirs("data/batch_results", exist_ok=True)

    with open(f"data/batch_results/{prefix}_{batch_idx}.jsonl", "wb") as f:
        f.write(raw_bytes)

    print("Saved:", emb_path)

    return prefix, emb_path, id_path


def batch_completed(batch_file, emb_path, id_path):

    if not os.path.exists(emb_path):
        return False

    if not os.path.exists(id_path):
        return False

    try:
        emb = np.load(emb_path)
        ids = np.load(id_path)

        with open(batch_file, "r", encoding="utf-8") as f:
            expected = sum(1 for _ in f)

        if len(emb) != expected:
            return False

        if len(ids) != expected:
            return False

        return True

    except:
        return False

def process_batch(prefix, batch_file, batch_idx, emb_dir):

    emb_path = os.path.join(emb_dir, f"chunk_{prefix}_{batch_idx}.npy")
    id_path = os.path.join(emb_dir, f"chunk_ids_{prefix}_{batch_idx}.npy")

    if batch_completed(batch_file, emb_path, id_path):
        print("Skipping completed batch:", batch_file)
        return None


    batch_id = submit_batch(batch_file)

    batch = wait_for_batch(batch_id)

    if batch.status == "failed":
        print("Batch failed:", batch_file)

        with open(batch_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print("Example failing input:")
        print(lines[0][:300])

        return None


    if batch.status == "completed":
        return download_embeddings(batch, emb_dir, prefix, batch_idx)

    

# --------------------------------------------------
# Prepare Jobs
# --------------------------------------------------

def prepare_jobs(table_a, table_b, emb_dir):

    jobs = []

    textsA = load_and_serialize(table_a)
    batch_dirA = os.path.join(emb_dir, "batches_tableA")
    batch_filesA = write_batch_files(textsA, batch_dirA)

    for i, batch_file in enumerate(batch_filesA):
        jobs.append(("tableA", batch_file, i))

    textsB = load_and_serialize(table_b)
    batch_dirB = os.path.join(emb_dir, "batches_tableB")
    batch_filesB = write_batch_files(textsB, batch_dirB)

    for i, batch_file in enumerate(batch_filesB):
        jobs.append(("tableB", batch_file, i))

    return jobs


# --------------------------------------------------
# Run jobs
# --------------------------------------------------

def run_all_batches(jobs, emb_dir):

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for prefix, batch_file, idx in jobs:

            futures.append(
                executor.submit(
                    process_batch,
                    prefix,
                    batch_file,
                    idx,
                    emb_dir
                )
            )

        for future in as_completed(futures):

            result = future.result()

            if result:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

    return results



# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):

    os.makedirs(args.emb_dir, exist_ok=True)
    os.makedirs(args.faiss_dir, exist_ok=True)

    jobs = prepare_jobs(
        args.table_a,
        args.table_b,
        args.emb_dir
    )

    print("Total batch jobs:", len(jobs))

    results = run_all_batches(
        jobs,
        args.emb_dir
    )


    print("Pipeline finished")


# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--table_a", required=True)
    parser.add_argument("--table_b", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--faiss_dir", required=True)

    args = parser.parse_args()

    main(args)