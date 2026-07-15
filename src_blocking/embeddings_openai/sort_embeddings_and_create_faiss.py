#!/usr/bin/env python3

import os
import glob
import re
import numpy as np
import faiss


EMB_DIR = "data/blocking_benchmark_final/embeddings/openai/large"
PREFIX = "tableB"

OUTPUT_EMB = os.path.join(EMB_DIR, f"{PREFIX}_embeddings.npy")
OUTPUT_IDS = os.path.join(EMB_DIR, f"{PREFIX}_ids.npy")
FAISS_PATH = os.path.join("data/blocking_benchmark_final/faiss", f"{PREFIX}.index")


def extract_number(path):
    name = os.path.basename(path)

    patterns = [
        rf"chunk_{PREFIX}_(\d+)\.npy$",
        rf"chunk_ids_{PREFIX}_(\d+)\.npy$",
        rf"chunk_{PREFIX}_retry_(\d+)\.npy$",
        rf"chunk_ids_{PREFIX}_retry_(\d+)\.npy$",
    ]

    for p in patterns:
        m = re.search(p, name)
        if m:
            num = int(m.group(1))
            return num

    raise ValueError(f"Unexpected filename: {name}")

def load_chunks():

    chunk_pattern = os.path.join(EMB_DIR, f"chunk_{PREFIX}_*.npy")
    id_pattern = os.path.join(EMB_DIR, f"chunk_ids_{PREFIX}_*.npy")

    chunk_paths = sorted(glob.glob(chunk_pattern), key=extract_number)
    id_paths = sorted(glob.glob(id_pattern), key=extract_number)

    if len(chunk_paths) == 0:
        raise RuntimeError("No embedding chunks found")

    if len(chunk_paths) != len(id_paths):
        raise RuntimeError("Mismatch between chunk files and id files")

    print("Found chunks:", len(chunk_paths))

    embeddings_list = []
    ids_list = []

    for cpath, ipath in zip(chunk_paths, id_paths):

        print("Loading:", os.path.basename(cpath))

        emb = np.load(cpath)
        ids = np.load(ipath)

        if emb.shape[0] != ids.shape[0]:
            raise RuntimeError(f"Size mismatch in {cpath}")

        embeddings_list.append(emb)
        ids_list.append(ids)

    embeddings = np.vstack(embeddings_list)
    ids = np.concatenate(ids_list)

    print("Total embeddings loaded:", embeddings.shape[0])

    return embeddings, ids


def sort_by_id(embeddings, ids):

    print("Sorting embeddings by ID...")

    order = np.argsort(ids)

    ids_sorted = ids[order]
    embeddings_sorted = embeddings[order]

    return embeddings_sorted, ids_sorted


def validate_row_equals_id(ids):

    print("Validating row == id ...")

    for i, val in enumerate(ids):
        if i != val:
            raise RuntimeError(f"Row-ID mismatch at row {i}, id={val}")

    print("Validation successful: row == id")


def build_faiss(embeddings):

    print("Building FAISS index")

    dim = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    print("Total indexed vectors:", index.ntotal)

    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

    faiss.write_index(index, FAISS_PATH)

    print("FAISS index written to:", FAISS_PATH)


def main():

    embeddings, ids = load_chunks()

    embeddings, ids = sort_by_id(embeddings, ids)

    validate_row_equals_id(ids)

    np.save(OUTPUT_EMB, embeddings)
    np.save(OUTPUT_IDS, ids)

    print("Saved embeddings:", OUTPUT_EMB)
    print("Saved ids:", OUTPUT_IDS)

    build_faiss(embeddings)


if __name__ == "__main__":
    main()