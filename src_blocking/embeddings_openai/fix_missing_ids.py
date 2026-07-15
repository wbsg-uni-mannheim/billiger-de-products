#!/usr/bin/env python3

import json
import time
import os
import numpy as np
import pandas as pd
from openai import OpenAI
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)
import html

client = OpenAI(api_key=API_key)

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

def process_batch(prefix, batch_file, batch_idx, emb_dir):

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
    

import glob
import json
import numpy as np

def check_retry_ids_present(retryA_file, retryB_file, emb_dir):

    expected_ids = set()
    embedded_ids = set()

    # --- collect expected ids from retry files ---
    retry_files = []

    if retryA_file and os.path.exists(retryA_file):
        print("Found retry file for tableA:", retryA_file)
        retry_files.append(retryA_file)

    if retryB_file and os.path.exists(retryB_file):
        print("Found retry file for tableB:", retryB_file)
        retry_files.append(retryB_file)

    for retry_file in retry_files:
        with open(retry_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                expected_ids.add(int(obj["custom_id"]))

    # --- collect ids only from retry embedding outputs ---
    retry_id_files = glob.glob(f"{emb_dir}/chunk_ids_*retry*.npy")

    for id_file in retry_id_files:
        ids = np.load(id_file)
        embedded_ids.update(ids.tolist())

    missing_ids = expected_ids - embedded_ids
    extra_ids = embedded_ids - expected_ids

    print("Retry expected IDs:", len(expected_ids))
    print("Retry embedded IDs:", len(embedded_ids))
    print("Missing retry IDs:", len(missing_ids))
    print("Unexpected retry IDs:", len(extra_ids))

    if missing_ids:
        print("Example missing retry IDs:", list(sorted(missing_ids))[:20])

    if extra_ids:
        print("Example unexpected retry IDs:", list(sorted(extra_ids))[:20])

    if not missing_ids:
        print("\nAll retry embeddings successfully generated.")

    return missing_ids



def shorten_retry_file(in_file, out_file, max_chars=7000):

    with open(in_file, "r", encoding="utf-8") as f_in, \
         open(out_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            obj = json.loads(line)

            text = obj["body"]["input"]

            # decode HTML entities (&uuml; → ü etc.)
            text = html.unescape(text)

            # normalize whitespace
            text = " ".join(text.split())

            # truncate safely
            text = text[:max_chars]

            obj["body"]["input"] = text

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Created shortened retry file:", out_file)

def run_embedding():
    emb_dir = "data/blocking_benchmark_final/embeddings/openai/large"

    retryA = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableA/retry_tableA.jsonl"
    retryB = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableB/retry_tableB.jsonl"

    if os.path.exists(retryA):
        print("Found retry file for tableA:", retryA)

        retryA_short = retryA.replace(".jsonl", "_short.jsonl")

        shorten_retry_file(retryA, retryA_short)

        process_batch("tableA_retry", retryA_short, 0, emb_dir)


    if os.path.exists(retryB):
        print("Found retry file for tableB:", retryB)

        retryB_short = retryB.replace(".jsonl", "_short.jsonl")

        shorten_retry_file(retryB, retryB_short)

        process_batch("tableB_retry", retryB_short, 0, emb_dir)

    retraA_results_path = f"data/batch_results/*"
    retraB_results_path = f"data/batch_results/*"

if __name__ == "__main__":
    emb_dir = "data/blocking_benchmark_final/embeddings/openai/large"

    retryA = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableA/retry_tableA.jsonl"
    retryB = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableB/retry_tableB.jsonl"

    retryA_short = retryA.replace(".jsonl", "_short.jsonl") if retryA else None
    retryB_short = retryB.replace(".jsonl", "_short.jsonl") if retryB else None

    run_embedding()

    check_retry_ids_present(
        retryA_file=retryA_short,
        retryB_file=retryB_short,
        emb_dir=emb_dir
    )
    