import json
import pandas as pd
import time
import gzip
import os
import argparse
from sklearn.metrics import f1_score
from openai import OpenAI

API_key = os.environ.get("OPENAI_API_KEY")

# =========================
# === CONFIG / CLIENT  ===
# =========================
client = OpenAI(api_key=API_key)

# Process record GPT prompt
def process_record(record):

    # Extract 'left' values (excluding ID)
    left_parts = [
        "Marke: "+ str(record.get("brand_left")) if record.get("brand_left") is not None else "" ,
        "Name: "+str(record.get("name_left")) if record.get("name_left") is not None else "" ,
        "Preis: "+str(record.get("price_left")) if record.get("price_left") is not None else "-" + "Euro",
        "Beschreibung: " + str(record.get("desc_left")) if record.get("desc_left") is not None else ""
    ]
    left_text = " ".join(filter(None, left_parts))  # filter out empty strings

    # Extract 'right' values (excluding ID)
    right_parts = [
        "Marke: "+ str(record.get("brand_right")) if record.get("brand_right") is not None else "",
        "Name: "+str(record.get("name_right")) if record.get("name_right") is not None else "",
        "Preis: "+str(record.get("price_right")) if record.get("price_right") is not None else "-" + "Euro",
        "Beschreibung: " +str(record.get("desc_right")) if record.get("desc_right") is not None else "" 
    ]
    right_text = " ".join(filter(None, right_parts))  # filter out empty strings

    label = int(record.get("label", -1))
    if label == -1:
        print("Warning: Label is missing or invalid, setting to -1")

    # Example of further processing (modify as required)
    left_text = left_text.replace("/", " ")  # Replace slashes with spaces
    right_text = right_text.replace("/", " ")  # Replace slashes with spaces

    return left_text, right_text, label

# ==================================
# === BUILD BATCH INPUT (.jsonl) ===
# ==================================

def build_batch_file(cc, un, batch_path):
    src = f"data/solute_de/gold-standards_adjusted/products{cc}rnd{un}un_gs.json.gz" #Adjusted Be Aware Here english dataset
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)

    sidecar = {}

    with gzip.open(src, "rt", encoding="utf-8") as infile, \
         open(batch_path, "w", encoding="utf-8") as outfile:

        for i, line in enumerate(infile):
            record = json.loads(line)
            e1, e2, label = process_record(record)

            prompt = (
                "Beziehen sich diese beiden Produktbeschreibungen auf dasselbe reale Produkt?\n"
                "Antworte nur mit Ja oder Nein.\n"
                f"Produkt 1: {e1}\n"
                f"Produkt 2: {e2}"
            )

            req = {
                "custom_id": str(record.get("pair_id")),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.gptmodel,
                    "messages": [{"role": "user", "content": prompt}],
                }
            }

            outfile.write(json.dumps(req, ensure_ascii=False) + "\n")

            sidecar[str(record.get("pair_id"))] = {
                "label": label,
                "entity_1": e1,
                "entity_2": e2,
                "is_hard_negative": int(record.get("is_hard_negative"))
            }

    meta_path = batch_path.replace(".jsonl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False)

    print("Batch input written to:", batch_path)
    print("Metadata written to:", meta_path)

# =========================
# === BATCH LIFECYCLE  ===
# =========================

def submit_batch(batch_path):
    file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
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
            print("Batch did not complete successfully.")
            # Dump the whole object so you can see *why*
            try:
                print(json.dumps(batch.model_dump(), indent=2))
            except Exception as e:
                print(batch, "\n", str(e))
            raise RuntimeError("Batch failed.")

        time.sleep(60)


def download_results(batch, out_path):
    content = client.files.content(batch.output_file_id)
    with open(out_path, "wb") as f:
        f.write(content.read())
    print("Results saved to:", out_path)

# =========================
# === PARSE RESULTS    ===
# =========================

def parse_results(result_path):
    rows = []
    meta_path = result_path.replace("data/batch_results", "data/batch_inputs").replace(".jsonl", "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_lookup = json.load(f)

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            

            pair_id = obj["custom_id"]
            meta = meta_lookup[pair_id]

            resp = obj["response"]["body"]

            answer = resp["choices"][0]["message"]["content"]
            if "ja" in answer.strip().lower() or answer.strip().lower() in ("ja", "1"):
                answer_int = 1
            elif "nein" in answer.strip().lower() or answer.strip().lower() in ("nein", "0"):
                answer_int = 0
            else:
                answer_int = -1  # Invalid or unrecognized answer

            match = int(answer_int == meta["label"])

            rows.append({
                "Entity1": meta["entity_1"],
                "Entity2": meta["entity_2"],
                "Pair_ID": pair_id,
                "Answer": answer,
                "Answer_binary": answer_int,
                "Label": meta["label"],
                "Match": match,
                "Hard_Negative": meta["is_hard_negative"]
            })

    return pd.DataFrame(rows)


# =========================
# === METRICS & TRACK ===
# =========================

def print_f1(cc, un, additional_naming=""):
    df = pd.read_csv(f"results/generated/gpt/de/{args.gptmodel}/csv_results/products_{cc}_{un}un_batched_german_{additional_naming}.csv")
    if (df["Answer_binary"] == -1).sum() > 0:
        print("Found -1 values in Answer_binary column")
        df = df[df["Answer_binary"] != -1]
    f1 = f1_score(df["Label"], df["Answer_binary"])
    print("F1:", f1)

    count_non_match = (df["Match"] == 0).sum()
    print("Wrongly matched by GPT: ", count_non_match)
    os.makedirs(f"results/generated/gpt/de/{args.gptmodel}/f1", exist_ok=True)
    with open(f"results/generated/gpt/de/{args.gptmodel}/f1/f1_score_{cc}_{un}un_german_{additional_naming}.txt", "w", encoding="utf-8") as f:
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Wrongly matched by GPT: {count_non_match}\n")

    return f1


def run_benchmark(additional_naming=""):



    batch_path = f"data/batch_inputs/gpt_de/{args.gptmodel}/{args.gptmodel}_{args.cc}_{args.un}_batched_german_{additional_naming}.jsonl"
    result_path = f"data/batch_results/gpt_de/{args.gptmodel}/{args.gptmodel}_{args.cc}_{args.un}_batched_german_{additional_naming}.jsonl"
    os.makedirs(f"data/batch_results/gpt_de/{args.gptmodel}", exist_ok=True)
    os.makedirs(f"data/batch_inputs/gpt_de/{args.gptmodel}", exist_ok=True)

    
    build_batch_file(args.cc, args.un, batch_path)
    batch_id = submit_batch(batch_path)
    batch = wait_for_batch(batch_id)
    download_results(batch, result_path)
    

    df = parse_results(result_path)

    os.makedirs(f"results/generated/gpt/de/{args.gptmodel}/csv_results", exist_ok=True)
    out_csv = f"results/generated/gpt/de/{args.gptmodel}/csv_results/products_{args.cc}_{args.un}un_batched_german_{additional_naming}.csv"
    df.to_csv(out_csv, index=False)

    print_f1(args.cc, args.un, additional_naming)


    

# Main execution with dynamic cc and un values
# =========================
# === MAIN             ===
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--un", required=True)
    parser.add_argument("--gptmodel", choices=("gpt-5.2",), default="gpt-5.2")
    args = parser.parse_args()

    # cc und un aus den Argumenten verwenden
    run_benchmark("easy_prompt")
# Run as:
# source .venv/bin/activate
# python src/models/gpt/gpt_batch_german.py --cc="80cc20" --un="050" --gptmodel="gpt-5.2"
