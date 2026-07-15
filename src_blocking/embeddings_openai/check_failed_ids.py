import json
import glob
import os
import re
import json
import glob
import os
import re

def collect_failed_and_missing_requests(batch_file, result_file):

    expected = {}
    success = set()
    explicit_failures = set()

    # read original batch requests
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = int(obj["custom_id"])
            expected[cid] = line

    # read result file
    with open(result_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = json.loads(line)
            cid = int(obj["custom_id"])

            if "response" not in obj:
                explicit_failures.add(cid)
                continue

            status = obj["response"].get("status_code")

            if status == 200:
                success.add(cid)
            else:
                explicit_failures.add(cid)

    missing = set(expected.keys()) - success - explicit_failures

    retry_ids = explicit_failures | missing

    return retry_ids, explicit_failures, missing, expected

def build_retry_files():

    retryA = []
    retryB = []

    batchA = glob.glob("data/blocking_benchmark_final/embeddings/openai/large/batches_tableA/*.jsonl")
    batchB = glob.glob("data/blocking_benchmark_final/embeddings/openai/large/batches_tableB/*.jsonl")

    for batch_file in batchA:

        batch_idx = re.search(r"_(\d+)\.jsonl$", batch_file).group(1)
        batch_idx = int(batch_idx)
        result_file = f"data/batch_results/tableA_{batch_idx}.jsonl"

        if not os.path.exists(result_file):
            print(f"Result file missing for {batch_file}: {result_file}")
            continue

        retry_ids, explicit_failures, missing, expected = collect_failed_and_missing_requests(
            batch_file, result_file
        )

        for cid in retry_ids:
            retryA.append(expected[cid])

    for batch_file in batchB:

        batch_idx = re.search(r"_(\d+)\.jsonl$", batch_file).group(1)
        batch_idx = int(batch_idx)
        result_file = f"data/batch_results/tableB_{batch_idx}.jsonl"

        if not os.path.exists(result_file):
            print(f"Result file missing for {batch_file}: {result_file}")
            continue

        retry_ids, explicit_failures, missing, expected = collect_failed_and_missing_requests(
            batch_file, result_file
        )

        for cid in retry_ids:
            retryB.append(expected[cid])

    # write retry files
    if retryA:
        with open("data/blocking_benchmark_final/embeddings/openai/large/batches_tableA/retry_tableA.jsonl", "w", encoding="utf-8") as f:
            f.writelines(retryA)

    if retryB:
        with open("data/blocking_benchmark_final/embeddings/openai/large/batches_tableB/retry_tableB.jsonl", "w", encoding="utf-8") as f:
            f.writelines(retryB)

    print("Retry requests for tableA:", len(retryA))
    print("Retry requests for tableB:", len(retryB))


def find_failed_ids(batch_file, result_file):

    expected_ids = set()
    returned_ids = set()
    failed_ids = {}

    # expected ids
    with open(batch_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            expected_ids.add(int(obj["custom_id"]))

    # results
    with open(result_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = json.loads(line)
            cid = int(obj["custom_id"])

            if "response" not in obj:
                failed_ids[cid] = "no response"
                continue

            status = obj["response"].get("status_code", None)

            if status != 200:
                error_msg = obj["response"]["body"].get("error", {}).get("message", "unknown error")
                failed_ids[cid] = error_msg
            else:
                returned_ids.add(cid)

    missing_ids = expected_ids - returned_ids - set(failed_ids.keys())

    print("Expected items:", len(expected_ids))
    print("Returned embeddings:", len(returned_ids))
    print("Explicit errors:", len(failed_ids))
    print("Missing ids:", len(missing_ids))

    if failed_ids:
        print("\nExample failures:")
        for cid, reason in list(failed_ids.items())[:10]:
            print(cid, "->", reason)

    return failed_ids, missing_ids

def missing_items():
    result_path = "data/batch_results/*"
    batch_pathA = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableA/*"
    batch_pathB = "data/blocking_benchmark_final/embeddings/openai/large/batches_tableB/*"

    for a_batch_file in glob.glob(batch_pathA):
        a_batch_name = os.path.basename(a_batch_file)
        batch_idx = re.search(r"batch_(\d+)\.jsonl$", a_batch_name).group(1)
        batch_idx = int(batch_idx)
        result_file = f"data/batch_results/tableA_{batch_idx}.jsonl"

        if not os.path.exists(result_file):
            print(f"Result file missing for {a_batch_file}: {result_file}")
            continue

        print(f"\nChecking batch {batch_idx} for tableA...")
        MISSING_IDS = find_failed_ids(a_batch_file, result_file)
    
    for b_batch_file in glob.glob(batch_pathB):
        b_batch_name = os.path.basename(b_batch_file)
        batch_idx = re.search(r"batch_(\d+)\.jsonl$", b_batch_name)
        if batch_idx is None:
            print(f"Could not extract batch index from filename: {b_batch_file}")
            continue
        batch_idx = batch_idx.group(1)
        batch_idx = int(batch_idx)
        result_file = f"data/batch_results/tableB_{batch_idx}.jsonl"

        if not os.path.exists(result_file):
            print(f"Result file missing for {b_batch_file}: {result_file}")
            continue

        print(f"\nChecking batch {batch_idx} for tableB...")
        MISSING_IDS = find_failed_ids(b_batch_file, result_file)

if __name__ == "__main__":
    
    missing_items()
    #build_retry_files()

