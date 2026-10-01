# Benchmark construction code

This folder contains the code that built the German product matching benchmark from the billiger.de offer corpus. It originates from the master thesis "Entity Matching Pipeline for German Product Data" by Ksenia Elagin and is published here to document the procedure.

The raw billiger.de corpus and the intermediate files (pickles, DBSCAN cluster tables, OpenAI batch files) are not released. The code therefore cannot be run end to end; it documents what was done. Notebook outputs were removed, the OpenAI API key is read from the `OPENAI_API_KEY` environment variable, and otherwise the code is unchanged. File paths are relative to the thesis working directory: the notebooks and `generate-sets-final.py` were run from a folder next to `data/` and use `../data/...`, the other scripts were run from the thesis root and use `data/...`.

## Pipeline

| Step | File | Input | Output |
| --- | --- | --- | --- |
| 1. Cleansing and filtering | `cleansing/data_cleaning.ipynb` | raw offer dump `data/raw/rev2_docs_since_2020_01_01.json` (6,309,224 offers), fastText language model `lid.176.bin` | `data/working/dedup_preprocessed_..._only_long_name.pkl.gz` |
| 2. DBSCAN clustering | `../notebooks/USB_code/dbscan_clustering.ipynb` (already in this repo, identical to the thesis version) | cleansed corpus from step 1 | `data/working/dbscan/{seen,unseen}_dbscan_{clusters,mapping}.csv` |
| 3. Product selection, splits, pairs | `set_generation/generate-sets-final.py` | cleansed corpus, DBSCAN tables, fastText product embeddings `deepmatcher_product_datasets.model` | `data/derived/{training-sets,validation-sets,gold-standards}/products*.json.gz` (pairwise and multi-class) |
| 4. Test label review | `label_review/error_analysis_excel_creation.py`, `label_review/adjust_testset.py` | gold standards, GPT matcher predictions, manually reviewed Excel sheets | `data/derived/gold-standards_adjusted/` |
| 5. English translation | `translation/transalte_datasets_to_english_multiple_batches.py` plus helpers | German pair files from steps 3 and 4 | `data/derived_en/...` (same structure, English text) |
| 6. Category mapping | `categories/category_extraction.py` | cleansed corpus (`shop_cat`, `name`) | corpus pickle with `top_category_mapped`, mapping and count CSVs |

### 1. Cleansing and filtering

`data_cleaning.ipynb` inspects the raw schema, then:

- keeps offers whose name plus description is classified as German by fastText `lid.176.bin` (German must be the top language among the top 5, and at most 4 distinct non-Latin script characters),
- drops offers without a name and removes duplicates on lowercased name + description + brand (for example the same offer re-crawled after a price update),
- keeps offers whose name has at least 5 tokens after lowercasing and punctuation stripping.

A cluster-level main-entity filter from the WDC Products code is present but commented out and was not applied.

### 2. DBSCAN clustering

Product clusters (`product_id`) with more than 3 offers are clustered by DBSCAN (cosine distance, eps 0.35, min_samples 1) on binary bag-of-words vectors of offer names. Clusters are written separately for seen candidates (7 to 80 offers per product) and unseen candidates (4 to 6 offers per product).

### 3. Product selection, splits, and pair generation

`generate-sets-final.py` (random seed 42):

- `select_cc_clusters` builds corner-case groups from DBSCAN clusters with at least 5 products: a random anchor plus 4 similar products, each added by a similarity metric drawn at random from cosine, generalized Jaccard (threshold 0.7), Dice, and fastText embedding similarity on the product name. Selections of 100, 250, and 400 corner-case products are nested.
- `select_rnd_clusters` adds random filler products (400, 250, 100), giving 500 products for the 20 %, 50 %, and 80 % corner-case variants.
- Seen, unseen (test), and unseen (validation) products are drawn from disjoint pools; asserts check nesting and disjointness.
- `build_pairs` assigns offers of each product to train, validation, and test. For corner-case products, the test and validation offers are taken from the least similar fifth of offer pairs.
- `build_train` and `build_test` create all positive pairs within a product. For each offer they add hard negatives (1, 2, or 3 for the small, medium, and large training sets; 3 in the test set), each the most similar offer of a different product under a randomly drawn name similarity, plus one random negative. Here the candidate list starts with SoftTF-IDF, so the drawn metric is SoftTF-IDF, cosine, generalized Jaccard, or Dice; fastText is in the list but never drawn. `generate_pairs` labels pairs and flags hard negatives.
- Test sets with 0 %, 50 %, and 100 % unseen products and small, medium, and large development sets are written for each variant (`80cc20rnd`, `50cc50rnd`, `20cc80rnd`).

### 4. Test label review

`error_analysis_excel_creation.py` exports test pairs where a GPT matcher disagreed with the label to Excel for manual review. `adjust_testset.py` applies the reviewed labels to the gold standards and drops pairs marked `-1`, producing `gold-standards_adjusted`.

### 5. English translation

`transalte_datasets_to_english_multiple_batches.py` collects every unique offer of the pair files, sends one OpenAI Batch API request per offer to `gpt-5-mini` (reasoning effort low) with the instruction to return valid JSON with identical keys, parses the results, logs parse errors, and writes English copies of the pair files. Helpers:

- `reprocess_incomplete_items.py`: re-submits offers whose batch response was incomplete,
- `repair.py`: mechanical repair of near-JSON outputs from the error log,
- `test_language.py`, `export_untranslated_ids.py`: fastText check for offers that were not translated and export of those offers,
- `test_how_much_shortened.py`: compares German and English token counts to detect shortened translations.

### 6. Category mapping

`category_extraction.py` maps the retailer category path (`shop_cat`) to 13 top-level categories in two stages:

1. rule-based: normalize the path, split it into segments, skip generic first-level segments, and count keyword hits per category (`HEURISTIC_RULES`, with simple German suffix matching); the category with most hits wins,
2. LLM: rows left as `Sonstige` or `Undefiniert` are classified by `gpt-5-mini`, once per distinct `shop_cat` (using one example product name), or per offer name when `shop_cat` is missing.

`error_analysis_category.py` draws the random sample of 500 corpus offers (seed 42) used for manual validation of the mapping. `category_distribution.py` joins the mapping to the pair files and computes the per-split category shares reported in the paper.
