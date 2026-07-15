import gzip
import json
import pandas as pd

def change_label(model, c, u):
    test_path = f"data/derived_en/gold-standards_adjusted/products{c}rnd{u}un_gs.json.gz"
    gpt_path = f"src/models/gpt/reports/{model}/csv_results/products_{c}_{u}un_batched_english_new_prompt.csv"

    change_label = {}
    all_pair_ids = set()
    # Adjust the labels in gpt path by using from test path 
    for line in gzip.open(test_path, "rt", encoding="utf-8"):
        record = json.loads(line)
        pair_id = record.get("pair_id")
        all_pair_ids.add(pair_id)
        label = record.get("label")
        change_label[pair_id] = label

    df = pd.read_csv(gpt_path)
    df["Label"] = df["Pair_ID"].map(change_label)
    #remove rows of pair ids which are not in the test set
    df = df[df["Pair_ID"].isin(all_pair_ids)]
    df.to_csv(f"src/models/gpt/reports/{model}/csv_results/products_{c}_{u}un_batched_english_new_prompt_adjusted.csv", index=False)

    #Give the f1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(df["Label"], df["Answer_binary"])
    print("Wrongly Matched: ", (df["Label"] != df["Answer_binary"]).sum())
    print("F1 after adjustment:", f1)
    with open(f"src/models/gpt/reports/{model}/f1/f1_score_{c}_{u}un_english_new_prompt.txt", "w", encoding="utf-8") as f:
        f.write("\nAdjusted Labels F1 Score\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Wrongly matched by GPT: {(df['Label'] != df['Answer_binary']).sum()}\n")

if __name__ == "__main__":
    model = "gpt-5.2"
    # Example usage
    cc = ["20cc80", "50cc50", "80cc20"]
    un = ["000","050", "100"]
    for c in cc:
        for u in un:
                change_label(model, c, u)