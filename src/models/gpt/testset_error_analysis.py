import json
import pandas as pd
import time
import gzip
import os
import argparse
import resource
from sklearn.metrics import f1_score
from codecarbon import OfflineEmissionsTracker
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)
from openai import OpenAI

# =========================
# === CONFIG / CLIENT  ===
# =========================
client = OpenAI(api_key=API_key)


# ==================================
# === BUILD BATCH INPUT (.jsonl) ===
# ==================================

def build_batch_file(cc, un, batch_path):
    src = f"src/models/gpt/reports/gpt-5-mini/products_{cc}_{un}un.csv"
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)

    sidecar = {}

    df = pd.read_csv(src)
    
    #Keep line where Match = 0
    df = df[df["Match"] == 0]

    with open(batch_path, "w", encoding="utf-8") as outfile:
        for i, row in df.iterrows():
            e1 = row["Entity1"]
            e2 = row["Entity2"]
            label = row["Label"]
            is_hard_negative = int(row["Hard_Negative"])
            # if pair ID does not exist
            pair_id = row["Pair_ID"]

            prompt = (
                "Handelt es sich bei diesen beiden Produkten um dasselbe reale Produkt?\n"
                f"Produkt 1: {e1}\n"
                f"Produkt 2: {e2}\n"
                "Sage Ja oder Nein und begründe kurz deine Antwort."
            )

            req = {
                "custom_id": pair_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.gptmodel,
                    "input": prompt,
                    "tools": [
                        { "type": "web_search" }
                    ]
                }}

            outfile.write(json.dumps(req, ensure_ascii=False) + "\n")

            sidecar[pair_id] = {
                "label": label,
                "entity_1": e1,
                "entity_2": e2,
                "is_hard_negative": is_hard_negative
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

            # --- Extract assistant message from Responses API output ---
            answer = ""
            urls = []

            for item in resp.get("output", []):
                if item.get("type") == "message" and item.get("role") == "assistant":
                    # Gather text
                    parts = [
                        chunk["text"]
                        for chunk in item.get("content", [])
                        if chunk.get("type") == "output_text"
                    ]
                    answer = "".join(parts)

                    # Gather URL citations
                    for ann in item.get("annotations", []):
                        if ann.get("type") == "url_citation":
                            uc = ann.get("url_citation", {})
                            urls.append({
                                "title": uc.get("title"),
                                "url": uc.get("url")
                            })
                    break

            usage = resp.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            a = answer.lower()
            if "ja" in a:
                answer_int = 1
            elif "nein" in a:
                answer_int = 0
            else:
                answer_int = -1

            match = int(answer_int == meta["label"])

            gpt_costs = float(
                (input_tokens / 1_000_000) * GPT_INPUT_COST +
                (output_tokens / 1_000_000) * GPT_OUTPUT_COST
            )

            rows.append({
                "Entity1": meta["entity_1"],
                "Entity2": meta["entity_2"],
                "Pair_ID": pair_id,
                "Answer": answer,
                "Answer_binary": answer_int,
                "Label": meta["label"],
                "Match": match,
                "Costs": gpt_costs,
                "Hard_Negative": meta["is_hard_negative"],
                "URLs": urls,  # list of {title, url}
            })

    return pd.DataFrame(rows)


# =========================
# === METRICS & TRACK ===
# =========================

def print_f1(cc, un):
    df = pd.read_csv(f"src/models/gpt/reports/{args.gptmodel}/products_{cc}_{un}un_erroranalysis_batched.csv")
    f1 = f1_score(df["Label"], df["Answer_binary"])
    print("F1:", f1)

    count_non_match = (df["Match"] == 1).sum()
    print("Correctly matched by GPT: ", count_non_match)
    
    with open(f"src/models/gpt/reports/{args.gptmodel}/f1_score_{cc}_{un}un.txt", "w", encoding="utf-8") as f:
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Wrongly matched by GPT: {count_non_match}\n")

    return f1


def run_with_tracking():
    os.makedirs("data/efficiency_tracker/gpt", exist_ok=True)
    csv_file = f"data/efficiency_tracker/gpt/{args.gptmodel}_cc{args.cc}_un{args.un}_erroranalysis_batched.csv"
    job_name = f"{args.gptmodel}_cc{args.cc}_un{args.un}_erroranalysis_batched"
    json_path = f"src/models/gpt/reports/{args.gptmodel}/efficiency_{args.cc}_{args.un}un_erroranalysis_batched.json"

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_file=f"data/efficiency_tracker/gpt/{args.gptmodel}_cc{args.cc}_un{args.un}_erroranalysis_batched.csv"
    )

    batch_path = f"data/batch_inputs/{args.gptmodel}_{args.cc}_{args.un}_erroranalysis_batched.jsonl"
    result_path = f"data/batch_results/{args.gptmodel}_{args.cc}_{args.un}_erroranalysis_batched.jsonl"
    os.makedirs("data/batch_results", exist_ok=True)

    start = time.time()
    tracker.start()
    
    build_batch_file(args.cc, args.un, batch_path)
    batch_id = submit_batch(batch_path)
    batch = wait_for_batch(batch_id)
    download_results(batch, result_path)
    
    tracker.stop()
    end = time.time()

    df = parse_results(result_path)

    os.makedirs(f"src/models/gpt/reports/{args.gptmodel}", exist_ok=True)
    out_csv = f"src/models/gpt/reports/{args.gptmodel}/products_{args.cc}_{args.un}un_erroranalysis_batched.csv"
    df.to_csv(out_csv, index=False)

    f1 = print_f1(args.cc, args.un)

    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    runtime = end - start

    # Calculate energy and costs
    emission_df = pd.read_csv(csv_file)
    energy_kwh = emission_df["energy_consumed"].iloc[-1]
    emissions_kg = emission_df["emissions"].iloc[-1]
    cost = pd.read_csv(out_csv)
    gpt_cost = cost["Costs"].sum()

    # Log result
    record = {
        "job_name": job_name,
        "runtime_sec": round(runtime, 3),
        "max_memory_mb": round(peak_ram_mb, 3), #peak_cpu_memory_mb
        "energy_kwh": round(energy_kwh, 6),
        "emissions_kg": round(emissions_kg, 6),
        "gpt_cost_eur": round(gpt_cost, 4),
        "f1_score": f1
    }

    # Append or create JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Runtime:", round(runtime, 2), "s")
    print("Peak RAM:", round(peak_ram_mb, 2), "MB")
    print("F1:", f1)

    

# Main execution with dynamic cc and un values
# =========================
# === MAIN             ===
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--un", required=True)
    parser.add_argument("--gptmodel", required=True)
    args = parser.parse_args()

    if args.gptmodel == "gpt-4o":
        GPT_INPUT_COST = 0.00015
        GPT_OUTPUT_COST = 0.00060
    elif args.gptmodel == "gpt-5.2":
        GPT_INPUT_COST = 0.025
        GPT_OUTPUT_COST = 2.00
    elif args.gptmodel == "gpt-5.2":
        # prices per 1M tokens
        GPT_INPUT_COST = 1.75
        GPT_OUTPUT_COST = 14.00
    else:
        raise ValueError("Unknown model for pricing")

    # cc und un aus den Argumenten verwenden
    run_with_tracking()
# Run as:
# source .venv/bin/activate
# python src/models/gpt/testse_error_analysis.py --cc="80cc20" --un="000" --gptmodel="gpt-5.2"