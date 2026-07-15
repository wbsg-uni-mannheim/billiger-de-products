import os
import pandas as pd

if __name__ == "__main__":
    model = "gpt-5.2"
    # Example usage
    cc = ["20cc80", "50cc50", "80cc20"]
    un = ["000","050", "100"]
    old_excels = pd.DataFrame()
    all_excels = pd.DataFrame()
    for c in cc:
        for u in un:
            excel_path = f"src/models/gpt/reports_old/{model}/reviewed_new_prompt/error_analysis_products_{c}_{u}un_english_new_prompt_5_2.xlsx"
            excel_path = pd.read_excel(excel_path).drop(columns=["Costs"])
            all_excels = pd.concat([all_excels, excel_path], ignore_index=True).drop_duplicates(subset="Pair_ID", keep="last")
           
    for c in cc:
        for u in un:
            excel_path_old = f"src/models/gpt/reports_old/{model}/reviewed/error_analysis_products_{c}_{u}un_filtered.xlsx"
            excel_path_old = pd.read_excel(excel_path_old).drop(columns=["Costs"])
            old_excels = pd.concat([old_excels, excel_path_old], ignore_index=True).drop_duplicates(subset="Pair_ID")
    
    
    #if old_excel contain pair_id that is not in all excels, concat those rows to all excels
    missing_rows = old_excels[~old_excels["Pair_ID"].isin(all_excels["Pair_ID"])]
    all_excels_new = pd.concat([all_excels, missing_rows], ignore_index=True).drop_duplicates(subset="Pair_ID", keep="first")
    print(f"Added {len(missing_rows)} missing rows from old excels to all excels. with these labels: {missing_rows['New Label'].value_counts().to_dict()}")
    

    all_excels_new.to_csv("src/models/gpt/error_analysis/all_excels.csv", index=False)
    all_excels_new.to_excel("src/models/gpt/error_analysis/all_excels.xlsx", index=False)

    #create a txt file of amount of changed labels, so Lale to New Label amount of changes
    #first create ductionary of Lale to New Label amount of changes
    label_changes = {}
    for index, row in all_excels_new.iterrows():
        old_label = row["Label"]
        new_label = row["New Label"]
        if old_label != new_label:
            if (old_label, new_label) not in label_changes:
                label_changes[(old_label, new_label)] = 0
            label_changes[(old_label, new_label)] += 1
    
    with open("src/models/gpt/error_analysis/label_changes.txt", "w") as f:
        for (old_label, new_label), count in label_changes.items():
            f.write(f"{old_label} to {new_label}: {count}\n")
        f.write(f"Total changed labels: {sum(label_changes.values())}\n")

    all_ids_changed_per_c_u = pd.DataFrame()
    for c in cc:
        for u in un:
            gold_path = f"data/derived/gold-standards/products{c}rnd{u}un_gs.json.gz"
            gold_df = pd.read_json(gold_path, compression="gzip", lines=True)
            gold_df = gold_df[["pair_id", "label"]]
            gold_df["c_u"] = f"{c}_{u}"
            # merge with rows in all excels that have the same pair_id, and get the new label
            merged_df = pd.merge(gold_df, all_excels_new[["Pair_ID", "New Label"]], left_on="pair_id", right_on="Pair_ID", how="inner")
            # keep merged_df rows where label != New Label
            changed_df = merged_df[merged_df["label"] != merged_df["New Label"]]
            changed_df = changed_df[["pair_id", "label", "New Label", "c_u"]]
            all_ids_changed_per_c_u = pd.concat([all_ids_changed_per_c_u, changed_df], ignore_index=True)

    all_ids_changed_per_c_u.to_csv("src/models/gpt/error_analysis/all_ids_changed_per_c_u.csv", index=False)
    #for each c_u combination get the (changed label pairs so label -> new label) and the amount of changes for each pair, and save to csv file
    label_changes_per_c_u = pd.DataFrame()
    for c_u, group in all_ids_changed_per_c_u.groupby("c_u"):
        label_changes = {}
        for index, row in group.iterrows():
            old_label = row["label"]
            new_label = row["New Label"]
            if (old_label, new_label) not in label_changes:
                label_changes[(old_label, new_label)] = 0
            label_changes[(old_label, new_label)] += 1
        for (old_label, new_label), count in label_changes.items():
            label_changes_per_c_u = label_changes_per_c_u.append({"c_u": c_u, "old_label": old_label, "new_label": new_label, "count": count}, ignore_index=True)

    label_changes_per_c_u.to_csv("src/models/gpt/error_analysis/label_changes_per_c_u.csv", index=False)
