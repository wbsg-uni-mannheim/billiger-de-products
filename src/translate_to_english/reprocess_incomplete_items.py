import json
import os
import time
from openai import OpenAI
import os; API_key = os.environ.get("OPENAI_API_KEY")  # set OPENAI_API_KEY env var (see REPRODUCTION.md)

client = OpenAI(api_key=API_key)
MAX_TRANSLATION_TOKENS = 4096

# -------------------------
# CONFIG – adjust paths
# -------------------------
folder = "training-sets"

combined_result_path = f"data/batch_results/translation/combined__{folder}.jsonl"
retry_batch_path = f"data/batch_inputs/translation/retry_incomplete__{folder}.jsonl"
retry_result_path = f"data/batch_results/translation/retry_incomplete__{folder}.jsonl"

# -------------------------
# Helpers
# -------------------------
def collect_incomplete_ids(batch_result_path):
    incomplete_ids = set()

    with open(batch_result_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("custom_id", "UNKNOWN")

            response = obj.get("response", None)
            if response is None:
                print(f"[{pid}] Missing response object")
                incomplete_ids.add(pid)
                continue

            body = response.get("body", None)
            if body is None:
                print(f"[{pid}] Body is None")
                incomplete_ids.add(pid)
                continue

            if not isinstance(body, dict):
                print(f"[{pid}] Body not dict: {type(body)}")
                incomplete_ids.add(pid)
                continue

            status = body.get("status", None)

            incomplete_details = body.get("incomplete_details") or {}
            incomplete_reason = incomplete_details.get("reason")

            if status == "incomplete":
                print(f"[{pid}] Incomplete (reason: {incomplete_reason})")
                incomplete_ids.add(pid)
                continue

            output = body.get("output") or []
            has_message = any(
                isinstance(o, dict) and o.get("type") == "message"
                for o in output
            )

            if not has_message:
                print(f"[{pid}] Missing message output")
                incomplete_ids.add(pid)

    print(f"Detected {len(incomplete_ids)} incomplete items.")
    return incomplete_ids


def load_original_batch(batch_input_path):
    requests = {}

    with open(batch_input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            requests[obj["custom_id"]] = obj

    return requests


def build_retry_file(incomplete_ids, original_requests, retry_batch_path):
    os.makedirs(os.path.dirname(retry_batch_path), exist_ok=True)

    with open(retry_batch_path, "w", encoding="utf-8") as f:
        for pid in incomplete_ids:
            if pid in original_requests:
                f.write(json.dumps(original_requests[pid], ensure_ascii=False) + "\n")

    print("Retry batch written:", retry_batch_path)


def submit_batch(batch_path):
    file_obj = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    print("Retry Batch ID:", batch.id)
    return batch.id


def wait_for_batch(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print("Status:", batch.status)

        if batch.status == "completed":
            return batch

        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Retry batch failed: {batch.status}")

        time.sleep(60)


def download_results(batch, out_path):
    if not batch.output_file_id:
        raise RuntimeError("Retry batch produced no output file.")

    content = client.files.content(batch.output_file_id)

    with open(out_path, "wb") as f:
        f.write(content.read())

    print("Retry results saved:", out_path)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    incomplete_ids = collect_incomplete_ids(combined_result_path)

    if not incomplete_ids:
        print("No incomplete items. Nothing to retry ✨")
        exit()

    original_batch_input = f"data/batch_inputs/translation/combined__{folder}.jsonl"
    original_requests = load_original_batch(original_batch_input)

    build_retry_file(incomplete_ids, original_requests, retry_batch_path)

    retry_batch_id = submit_batch(retry_batch_path)
    retry_batch = wait_for_batch(retry_batch_id)

    download_results(retry_batch, retry_result_path)

    print("Recovery run finished 🚀")
