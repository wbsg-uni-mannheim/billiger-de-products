#!/usr/bin/env python3
import argparse
import json
import time
import os
import numpy as np
import pandas as pd
import faiss
import glob
import re
from openai import OpenAI
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)

# -------------------------------
# OpenAI Client
# -------------------------------
client = OpenAI(api_key=API_key)

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


def write_batch_files(texts, out_dir, chunk_size=50000):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(0, len(texts), chunk_size):
        path = os.path.join(out_dir, f"batch_{i//chunk_size:04d}.jsonl")
        if os.path.exists(path):
            print(f"Batch file {path} already exists. Skipping creation.")
            paths.append(path)
            continue

        with open(path, "w", encoding="utf-8") as f:
            for j, text in enumerate(texts[i:i+chunk_size]):
                f.write(json.dumps({
                    "custom_id": str(i + j),  # GLOBAL ID!
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": text
                    }
                }) + "\n")
        paths.append(path)

    return paths



# -------------------------------
# OpenAI Batch Helpers
# -------------------------------
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

    print(f"Batch submitted: {batch.id}")
    return batch.id


def wait_for_batch(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")

        if batch.status == "completed":
            return batch

        if batch.status in ("failed", "expired", "cancelled"):
            # Erstmal Status klar anzeigen
            print(f"Batch {batch_id} failed with status: {batch.status}")

            # Falls OpenAI eine Error-Datei angelegt hat → auslesen
            if batch.error_file_id:
                print("Error file id:", batch.error_file_id)
                try:
                    error_content = client.files.content(batch.error_file_id)
                    print("=== ERROR DETAILS FROM OPENAI ===")
                    print(error_content.decode("utf-8"))
                except Exception as e:
                    print("Could not fetch error file:", e)
            else:
                print("No error_file_id provided by OpenAI.")

            raise RuntimeError(f"Batch {batch_id} failed")

        time.sleep(60)

def download_embeddings_with_ids(batch, out_path, prefix, i):
    response = client.files.content(batch.output_file_id)
    raw_bytes = response.read()
    lines = raw_bytes.decode("utf-8").splitlines()

    records = []
    ids = []
    for line in lines:
        obj = json.loads(line)

        idx = int(obj["custom_id"])
        emb = obj["response"]["body"]["data"][0]["embedding"]

        records.append(emb)
        ids.append(idx)

    X = np.array(records, dtype=np.float32)
    np.save(out_path, X)
    #save outputfile of openai
    os.makedirs("data/batch_results/embeddings", exist_ok=True)
    with open(f"data/batch_results/embeddings/{prefix}_{i}.jsonl", "wb") as f:
        f.write(response.read())

    print(f"Saved embeddings to {out_path} | shape={X.shape}")
    return X, np.array(ids, dtype=np.int32)

def process_table(csv_path, emb_dir, prefix, start_batch=0):
    texts = load_and_serialize(csv_path)
    batch_dir = os.path.join(emb_dir, f"batches_{prefix}")
    batch_files = write_batch_files(texts, batch_dir)

    all_chunk_paths = []  # Store paths to chunk files
    all_id_paths = []  # Store paths to ID files
    for i, batch_file in enumerate(batch_files):
        if i < start_batch:
            print(f"Skipping already processed batch {i}")
            continue

        batch_id = submit_batch(batch_file)
        batch = wait_for_batch(batch_id)
        outpath = os.path.join(emb_dir, f"chunk_{prefix}_{i}.npy")
        id_path = os.path.join(emb_dir, f"chunk_ids_{prefix}_{i}.npy")

        embeddings, ids = download_embeddings_with_ids(batch, outpath, prefix,i)
        np.save(id_path, ids)

        all_chunk_paths.append(outpath)
        all_id_paths.append(id_path)

    # Memory-efficient concatenation of chunks
    final_embeddings_path = os.path.join(emb_dir, f"{prefix}_embeddings.npy")
    final_ids_path = os.path.join(emb_dir, f"{prefix}_ids.npy")

    # Proper concatenation instead of raw binary append
    print("Assembling final embeddings array properly...")

    all_embeddings = []
    all_ids = []

    for chunk_path, id_path in zip(all_chunk_paths, all_id_paths):
        all_embeddings.append(np.load(chunk_path))
        all_ids.append(np.load(id_path))

    X = np.vstack(all_embeddings)
    ids = np.concatenate(all_ids)

    np.save(final_embeddings_path, X)
    np.save(final_ids_path, ids)

    print(f"Final embeddings saved to {final_embeddings_path} | shape={X.shape}")
    print(f"Final IDs saved to {final_ids_path} | shape={ids.shape}")

    # Load embeddings and IDs in smaller batches for FAISS indexing
    build_faiss_index_with_ids(all_chunk_paths, all_id_paths, os.path.join(args.faiss_dir, f"{prefix}.index"))
    return final_embeddings_path, final_ids_path

def rebuild_faiss_from_existing_chunks(emb_dir, prefix, out_index):
    print("=== Rebuilding FAISS from existing chunks ===")

    # find all embedding chunk files
    chunk_pattern = os.path.join(emb_dir, f"chunk_{prefix}_*.npy")
    id_pattern = os.path.join(emb_dir, f"chunk_ids_{prefix}_*.npy")

    chunk_paths = glob.glob(chunk_pattern)
    id_paths = glob.glob(id_pattern)

    if not chunk_paths:
        raise RuntimeError("No embedding chunks found.")

    # sort numerically by batch index
    def extract_number(path):
        return int(re.search(r"_(\d+)\.npy$", path).group(1))

    chunk_paths = sorted(chunk_paths, key=extract_number)
    id_paths = sorted(id_paths, key=extract_number)

    if len(chunk_paths) != len(id_paths):
        raise RuntimeError("Mismatch between embedding chunks and ID chunks.")

    index = None
    total_vectors = 0

    for chunk_path, id_path in zip(chunk_paths, id_paths):
        print(f"Loading {os.path.basename(chunk_path)}")

        embeddings = np.load(chunk_path)
        ids = np.load(id_path)

        if embeddings.shape[0] != ids.shape[0]:
            raise RuntimeError(f"Size mismatch in {chunk_path}")

        faiss.normalize_L2(embeddings)

        if index is None:
            dim = embeddings.shape[1]
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        index.add_with_ids(embeddings, ids)
        total_vectors += embeddings.shape[0]

        print(f"  → added {embeddings.shape[0]} vectors (total={total_vectors})")

    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    faiss.write_index(index, out_index)

    print("FAISS index written to:", out_index)
    print("Total indexed vectors:", total_vectors)

def build_faiss_index_with_ids(chunk_paths, id_paths, out_index):
    index = None

    for chunk_path, id_path in zip(chunk_paths, id_paths):
        embeddings = np.load(chunk_path)
        ids = np.load(id_path)
        faiss.normalize_L2(embeddings)

        if index is None:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        index.add_with_ids(embeddings, ids)

    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    faiss.write_index(index, out_index)

    print(f"FAISS index written to {out_index}")
    print(f"Indexed vectors: {index.ntotal}")


# -------------------------------
# Main Pipeline
# -------------------------------
def main(args):
    os.makedirs(args.emb_dir, exist_ok=True)
    os.makedirs(args.faiss_dir, exist_ok=True)

    print("=== Processing Table A ===")
    #process_table(args.table_a, args.emb_dir, "tableA")

    print("=== Processing Table B ===")
    #process_table(args.table_b, args.emb_dir, "tableB", start_batch=307)

    print("=== Building FAISS Index (Table B) ===")
    rebuild_faiss_from_existing_chunks(
        args.emb_dir,
        "tableB",
        os.path.join(args.faiss_dir, "tableB.index")
    )

    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_a", required=True)
    parser.add_argument("--table_b", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--faiss_dir", required=True)
    args = parser.parse_args()

    main(args)
