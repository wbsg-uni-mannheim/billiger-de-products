import pandas as pd
import glob
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os


def parse_dataset_name_rsup_en(name, efficiencytracker):
    if efficiencytracker:
        """
        roberta_predict_000un_large_lspc_id=0
        """

        m = re.search(
            r"^([^_]+)_predict_(\d+)un_(small|medium|large)_lspc_id=(\d+)$",
            name
        )
        if not m:
            raise ValueError(f"Cannot parse: {name}")

        test_unseen = m.group(2)
        train_size = m.group(3).capitalize()
        run_id = int(m.group(4))
        lm_model = m.group(1)

        return {
            "Test Unseen": f"{test_unseen}%",
            "Train Size": train_size,
            "LM Model": lm_model,
            "run": run_id,
        }
    else:
        """
        products20cc80rnd000un-large-all1024-5e-05-0.07-False-roberta-base
        """

        m = re.search(
            r"products(\d+)cc(\d+)rnd(\d+)un-(small|medium|large)-all(\d+)-([0-9.e-]+)-([0-9.]+)-([A-Za-z]+)-([A-Za-z-]+)",
            name
        )
        if not m:
            raise ValueError(f"Cannot parse: {name}")

        cc =  m.group(1)
        train_size = m.group(4).capitalize()
        lm_model = m.group(8)

        return {
            "Corner Cases": f"{cc}%",
            "Train Size": train_size,
            "LM Model": lm_model,
        }


def transformer_results_analysis_en():
    all_merged = []


    # ------------------------------------------------------------------
    # Load CodeCarbon logs
    # ------------------------------------------------------------------
    cc_records = []
    for jf in glob.glob(f"data/efficiency_tracker/roberta_en/*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            #only keep the last three rows
            data = data[-3:]

        for i,rec in enumerate(data):
            
            #First record is CC 80, second is CC50 und third is CC20
            if i == 0:
                cc = "80"
            elif i == 1:
                cc = "50"
            elif i == 2:
                cc = "20"
            else:
                raise ValueError(f"Unexpected record index: {i} in file {jf}")

            job = rec["job_name"]
            meta = parse_dataset_name_rsup_en(job, True)
            cc_records.append({
                **meta,
                "Corner Cases": f"{cc}%",
                "energy_kwh": rec["energy_kwh"],
                "emissions_kg": rec["emissions_kg"],
                "runtime_sec": rec["runtime_sec"],
                "max_memory_mb": rec["max_memory_mb"],
            })

    carbon_df = pd.DataFrame(cc_records)

    # ------------------------------------------------------------------
    # Load F1 results
    # ------------------------------------------------------------------
    perf_rows = []

    
    for folder in glob.glob(f"src/models/transformer_bert_confidence/reports_en/baseline/*"):
        if folder.__contains__("roberta-base"):
            for id_folder in glob.glob(f"{folder}/*"):
                name = Path(folder).name
                meta = parse_dataset_name_rsup_en(name, False)
                run_id = Path(id_folder).name
                if not run_id.isdigit():
                    continue  # no info there
                #add run id to meta
                meta["run"] = int(run_id)
                unseens = ["", "_un050", "_un100"]

                results_path = Path(id_folder) / "all_results.json"
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for unseen in unseens:
                        f1 = data.get(f"predict{unseen}_f1")
                        recall = data.get(f"predict{unseen}_recall")
                        precision = data.get(f"predict{unseen}_precision")
                        accuracy = data.get(f"predict{unseen}_accuracy")
                        if unseen == "":
                            test_unseen = "000"
                        elif unseen == "_un050":
                            test_unseen = "050"
                        elif unseen == "_un100":
                            test_unseen = "100"
                        perf_rows.append({
                            **meta,
                            "Test Unseen": f"{test_unseen}%",
                            "f1": f1,
                            "recall": recall,
                            "precision": precision,
                            "accuracy": accuracy,
                        })

    perf_df = pd.DataFrame(perf_rows)
    merged = perf_df.merge(
        carbon_df,
        on=["Corner Cases", "Train Size", "Test Unseen", "LM Model", "run"],
        how="left"
    )

    # ------------------------------------------------------------------------------
    # Group & compute mean + std (Abweichung)
    # ------------------------------------------------------------------------------

    summary = (
        merged
        .groupby([
            "Corner Cases",
            "Train Size",
            "Test Unseen",
            "LM Model"
        ])
        .agg(

            Precision_mean=("precision", "mean"),
            Precision_std=("precision", "std"),

            Recall_mean=("recall", "mean"),
            Recall_std=("recall", "std"),

            F1_mean=("f1", "mean"),
            F1_std=("f1", "std"),

            Energy_kWh_mean=("energy_kwh", "mean"),
            Energy_kWh_std=("energy_kwh", "std"),

            CO2_kg_mean=("emissions_kg", "mean"),
            CO2_kg_std=("emissions_kg", "std"),

            Runtime_mean=("runtime_sec", "mean"),
            Memory_MB_mean=("max_memory_mb", "mean")
        )
        .reset_index()
    )


    # ------------------------------------------------------------------------------
    # Efficiency metrics
    # ------------------------------------------------------------------------------

    summary["F1_per_kWh"] = summary["F1_mean"] / summary["Energy_kWh_mean"] 
    summary["F1_per_kgCO2"] = summary["F1_mean"] / summary["CO2_kg_mean"] 
    summary["F1_per_runtime"] = summary["F1_mean"] / summary["Runtime_mean"] 
    summary["F1_per_memory"] = summary["F1_mean"] / summary["Memory_MB_mean"] 
    cost_cols = ["Energy_kWh_mean", "CO2_kg_mean", "Runtime_mean", "Memory_MB_mean"]

    # min-max normalize costs per column to [0,1] (lower is better)
    for c in cost_cols:
        mn, mx = summary[c].min(), summary[c].max()
        summary[c + "_norm"] = (summary[c] - mn) / (mx - mn + 1e-12)

    summary["Cost_score"] = sum(summary[c + "_norm"] for c in cost_cols)  # equal weight
    summary["Efficiency_Score"] = summary["F1_mean"] / (summary["Cost_score"] + 1e-12)





    # ------------------------------------------------------------------------------
    # Save Excel with raw + summary
    # ------------------------------------------------------------------------------

    out = Path("src/models/transformer_bert_confidence/reports_en/analysis/roberta_experiment_summary_en.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print(f"Saved Excel report to {out}")


def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    column_path = "data/processed_en/gold-standards_adjusted"
    predictions_path = "src/models/transformer_bert_confidence/reports_en/predictions/*"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for folder in glob.glob(predictions_path):
        for folder_id in glob.glob(f"{folder}/*"):
            for file in glob.glob(f"{folder_id}/*.csv"):
                if not "baseline_predictions.csv" in file:
                    continue
                df_predictions = pd.read_csv(file)
    
                name = folder.split('/')[-1]
                gs_name = name.split("-")[0]
                cc = re.search(r"\d+cc\d+rnd", gs_name).group()
                un = re.search(r"\d+un", gs_name).group()
                train_size = name.split("-")[1]
                run_id = folder_id.split('/')[-1]
                #get from run0 to 0 from run_id
                run_id = run_id.replace("run", "")

                print("Processing Testset:", gs_name, " and Trainset:", train_size, " run:", run_id)
                gs_file =f"{column_path}/preprocessed_{gs_name}_gs.pkl.gz"

                df_columns = pd.read_pickle(gs_file, compression="gzip")
                #merge on rows and not on any column, because there is no common column between the two dataframes, but they are in the same order and have the same number of rows
                df_merged = pd.concat([df_predictions, df_columns[["id_right", "id_left"]]], axis=1)
                
                # Merge for id_left -> shop_cat_left 
                df_merged_left = pd.merge(df_merged, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
                
                # Rename shop_cat column correctly 
                df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
                
                # Merge for id_right -> shop_cat_right 
                df_merged_both_test = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
                
                # Rename shop_cat column from right merge 
                df_merged_both_test.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 
                
                df_merged_both_test["category_pair"] = (
                    df_merged_both_test[["shop_cat_left", "shop_cat_right"]]
                        .astype(str)
                        .apply(lambda x: " | ".join(sorted(x)), axis=1)
                )


                #get amount of items grouped by category from df_merged_both_test
                category_pair_counts = (
                    df_merged_both_test["category_pair"]
                        .value_counts(dropna=False)
                        .to_dict()
                )

                if "small" in train_size:
                    size = "small"
                elif "medium" in train_size:
                    size = "medium"
                elif "large" in train_size:
                    size = "large"
                
                # create new row in df merged that checks wether label == pred_label
                df_merged_both_test["match"] = df_merged_both_test["label"] == df_merged_both_test["prediction"]

                #create csv that goes through all categories that are in left and right and per category get percentage of wrong predictions (match == False) and total count of items in that category
                # Create a csv with columns: category, wrong_percentage, total_count
                
                for cat_pair, count in category_pair_counts.items():

                    cat_df = df_merged_both_test[df_merged_both_test["category_pair"] == cat_pair]

                    y_true = cat_df["label"]
                    y_pred = cat_df["prediction"]

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true,
                        y_pred,
                        average="binary",
                        zero_division=0
                    )

                    category_analysis.append({
                        "category_pair": cat_pair,
                        "f1": f1,
                        "total_count": len(cat_df),
                        "cc": cc,
                        "un": un,
                        "size": size,
                        "run_id": run_id
                    })
    
    #create datafram for grouped cc, un, size
    category_analysis_df = pd.DataFrame(category_analysis)
    os.makedirs("src/models/transformer_bert_confidence/reports_en/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv("src/models/transformer_bert_confidence/reports_en/analysis/categories/category_analysis_en.csv", index=False)

    #create an excel file that gets the average of f1 per category for each group of cc, un, size and model
    category_summary = (
        category_analysis_df
        .groupby(["category_pair", "cc", "un", "size"])
        .agg(
            f1_mean=("f1", "mean"),
            total_count_mean=("total_count", "mean")
        )
        .reset_index()
    )
    category_summary.to_excel("src/models/transformer_bert_confidence/reports_en/analysis/categories/category_summary_en.xlsx", index=False)


if __name__ == "__main__":
    #transformer_results_analysis_en()
    categories()
