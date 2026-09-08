# Benchmark construction: product-name clustering

The paper's DBSCAN step is documented in the [original construction notebook](../notebooks/USB_code/dbscan_clustering.ipynb). This is a source-only copy from repository commit `62007877d0ae51bc6f048a7bcd50182decc65aca`, with saved outputs and execution counts removed. The notebook operates on the original source corpus, which is not included in the public benchmark release.

The notebook specifies:

- One representative offer name per product, retaining products with at least four offers before clustering.
- Lowercasing, HTML-tag removal, punctuation removal, and whitespace normalization.
- Binary token vectors using `CountVectorizer(strip_accents='unicode', binary=True, min_df=4)`.
- `DBSCAN(metric='cosine', eps=0.35, min_samples=1)`.
- Candidate pools containing products with 7–80 offers (Seen) and 4–6 offers (Unseen), retaining candidate clusters with more than two rows within each pool.

DBSCAN provides coarse candidate groups for the subsequent corner-case selection. It does not determine match labels. The released pairs and the model-input preparation described in [REPRODUCTION.md](../REPRODUCTION.md) are the entry point for reproducing matching results.
