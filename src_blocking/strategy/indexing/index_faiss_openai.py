import argparse
import numpy as np
import faiss
import os


def main(args):
    X = np.load(args.embedding_file).astype("float32")

    if args.metric == "cosine":
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
    else:
        index = faiss.IndexFlatL2(X.shape[1])

    index.add(X)

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    faiss.write_index(index, args.out_index)

    print(f"FAISS index written to {args.out_index}")
    print("Vectors:", index.ntotal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_file", required=True)
    parser.add_argument("--out_index", required=True)
    parser.add_argument("--metric", default="cosine")
    args = parser.parse_args()
    main(args)
