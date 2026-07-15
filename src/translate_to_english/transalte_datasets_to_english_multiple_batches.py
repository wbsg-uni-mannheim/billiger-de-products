import json
import pandas as pd
import time
import gzip
import os
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)
from openai import OpenAI

# =========================
# === CONFIG / CLIENT ===
# =========================
client = OpenAI(api_key=API_key)
MAX_TRANSLATION_TOKENS = 4096


def collect_unique_products(dataset_path):
    products = {}
    with gzip.open(dataset_path, "rt", encoding="utf-8") as infile:
        for line in infile:
            r = json.loads(line)

            for side in ("left", "right"):
                id = str(r[f"id_{side}"])
                if id not in products:
                    products[id] = {
                        "product_id": id,
                        "name": r.get(f"name_{side}"),
                        "desc": r.get(f"desc_{side}")
                    }
    return products

def build_product_batch_file(products, batch_path):
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)

    with open(batch_path, "w", encoding="utf-8") as outfile:
        for pid, p in products.items():

            payload = {
                "name": p["name"],
                "desc": p["desc"]
            }
            req = {
                "custom_id": str(pid),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5-mini",
                    "input": (
                        "You are an expert at translating product information from German to English.\n"
                        "Translate the following JSON fields into English.\n"
                        "Return ONLY valid JSON with identical keys.\n\n"
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                    "reasoning": {"effort": "low"},
                    "max_output_tokens": MAX_TRANSLATION_TOKENS,
                }
            }
            outfile.write(json.dumps(req, ensure_ascii=False) + "\n")
    print("Batch input written:", batch_path)

# =========================
# === BATCH LIFECYCLE ===
# =========================
def submit_batch(batch_path):
    file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    print("Batch ID:", batch.id)
    return batch.id

def wait_for_batch(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print("Status:", batch.status)
        if batch.status == "completed":
            return batch
        if batch.status in ("failed", "expired", "cancelled"):
            # Dump the whole object so you can see *why*
            try:
                print(json.dumps(batch.model_dump(), indent=2))
            except Exception as e:
                print(batch, "\n", str(e))
            raise RuntimeError("Batch failed.")
        time.sleep(60)

def download_results(batch, out_path):
    if not batch.output_file_id:
        raise RuntimeError(
            f"No output file produced. error_file_id={batch.error_file_id}"
        )
    content = client.files.content(batch.output_file_id)
    with open(out_path, "wb") as f:
        f.write(content.read())
    print("Results saved to:", out_path)

# =========================
# === PARSE RESULTS ===
# =========================
def safe_parse_json(content, pair_id, errors, max_depth=3):
    """
    Try to parse JSON. On failure, record error and return None instead of crashing.
    """
    current = content
    for depth in range(max_depth):
        if isinstance(current, dict):
            return current
        if isinstance(current, str):
            current = current.strip()
            if not current.startswith("{"):
                errors.append({
                    "pair_id": pair_id,
                    "error": "Output does not start with '{'",
                    "raw_output": current
                })
                return None
            try:
                current = json.loads(current)
                continue
            except json.JSONDecodeError as e:
                errors.append({
                    "pair_id": pair_id,
                    "error": f"JSON decode error at depth {depth}: {e}",
                    "raw_output": current
                })
                return None
    errors.append({
        "pair_id": pair_id,
        "error": "Exceeded max JSON unwrap depth",
        "raw_output": str(content)
    })
    return None

def load_product_translations(batch_result_path, error_log_path):
    translations = {}
    errors = []

    usage_summary = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "items": 0
    }

    with open(batch_result_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["custom_id"]
            output = obj["response"]["body"]["output"]

            # Safely get the message content
            try:
                message = next(
                    item for item in output
                    if item.get("type") == "message"
                )
                content = message["content"][0]["text"]
            except StopIteration:
                errors.append({
                    "pair_id": pid,
                    "error": "No message of type 'message' found in output",
                    "raw_output": output
                })
                continue
            #cost tracking
            usage = obj["response"]["body"].get("usage", {})

            usage_summary["input_tokens"] += usage.get("input_tokens", 0)
            usage_summary["output_tokens"] += usage.get("output_tokens", 0)
            usage_summary["total_tokens"] += usage.get("total_tokens", 0)
            usage_summary["items"] += 1

            parsed = safe_parse_json(content, pid, errors)
            if parsed:
                translations[pid] = parsed

    if errors:
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, indent=2, ensure_ascii=False)

    return translations, usage_summary


def apply_product_translations(
    original_dataset_path,
    product_translations,
    output_dataset_path
):
    os.makedirs(os.path.dirname(output_dataset_path), exist_ok=True)

    with gzip.open(original_dataset_path, "rt", encoding="utf-8") as infile, \
         gzip.open(output_dataset_path, "wt", encoding="utf-8") as outfile:

        for line in infile:
            r = json.loads(line)

            left_id = str(r["id_left"])
            right_id = str(r["id_right"])

            if left_id in product_translations:
                t = product_translations[left_id]
                r["name_left"] = t["name"]
                r["desc_left"] = t["desc"]

            if right_id in product_translations:
                t = product_translations[right_id]
                r["name_right"] = t["name"]
                r["desc_right"] = t["desc"]

            outfile.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Translated dataset written:", output_dataset_path)

def compute_costs(usage, pricing):
    input_cost = (usage["input_tokens"] / 1000) * pricing["input_per_1k"]
    output_cost = (usage["output_tokens"] / 1000) * pricing["output_per_1k"]

    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6)
    }
def save_cost_report(path, dataset_name, usage, costs):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    report = {
        "dataset": dataset_name,
        "usage": usage,
        "costs_usd": costs
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
# Main execution with dynamic cc and un values
# =========================
# === MAIN ===
# =========================
if __name__ == "__main__":
    
    GPT5_MINI_BATCH_PRICING = {
        "input_per_1k": 0.00013,   # $ / 1K input tokens
        "output_per_1k": 0.00105   # $ / 1K output tokens
    }

    folder = "training-sets"
    #split = "20cc80"
    base_path = f"data/derived/{folder}"

    datasets = [
        f for f in os.listdir(base_path)
        if "multi" not in f.lower()
    ]

    print("Amount of datasets to process:", len(datasets))
    all_products = {}

    for dataset in datasets:
        src = os.path.join(base_path, dataset)
        dataset_name = dataset.replace(".json.gz", "")

        # Collect unique products
        products = collect_unique_products(src)
        print(f"{dataset}: {len(products)} products found.")

        all_products.update(products)
    
    print(f"Total amount products: {len(all_products)}")

    # Create a single batch for all products
    #combined_batch_path = f"data/batch_inputs/translation/combined_{folder}.jsonl"
    #build_product_batch_file(all_products, combined_batch_path)

    # Send and wait for the batch
    #batch_id = submit_batch(combined_batch_path)
    #batch = wait_for_batch(batch_id)

    # Download results
    combined_result_path = f"data/batch_results/translation/combined_{folder}.jsonl"
    #download_results(batch, combined_result_path)

    # Load translations
    error_log_path = f"data/batch_results/translation_errors/combined_{folder}_errors.json"
    product_translations, token_usage = load_product_translations(
        combined_result_path, error_log_path
    )
    print(f"Translated products: {len(product_translations)}")

    costs = compute_costs(token_usage, GPT5_MINI_BATCH_PRICING)
    save_cost_report(
        f"data/cost_reports/combined_{folder}_translation_cost.json",
        f"combined_{folder}",
        token_usage,
        costs
    )
    print("Estimate Costs (USD):", costs)

    # 6️⃣ Apply translations to all datasets
    for dataset in datasets:
        src = os.path.join(base_path, dataset)
        os.makedirs(f"data/derived_en/{folder}", exist_ok=True)
        out_path = f"data/derived_en/{folder}/{dataset}"

        apply_product_translations(
            src,
            product_translations,
            out_path
        )

    print("All datasets processed.")