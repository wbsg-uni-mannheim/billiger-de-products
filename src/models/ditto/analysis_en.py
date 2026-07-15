import pandas as pd
import glob
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)
import os

def parse_dataset_name(name, pred=False):
    if pred:  

        """
        "final_large_20cc80rnd000un_lm=roberta_da=del_dk=None_su=False_size=None_id=0_english_predictions_050"
        """

        m = re.search(
            r"final_(small|medium|large)_(\d+)cc\d+rnd(\d+)un_lm=([^_]+).*?_id=(\d+)_english_predictions_(\d+)",
            name
        )
        if not m:
            m = re.search(
                r"final_(small|medium|large)_(\d+)cc\d+rnd(\d+)un_lm=([^_]+).*?_id=(\d+)_english_predictions",
                name
            )
            test_unseen = m.group(3)
            if not m:
                raise ValueError(f"Cannot parse: {name}")
        else:    
            test_unseen = m.group(6)
        
        train_size = m.group(1).capitalize()
        cc = int(m.group(2))
        lm_model = m.group(4)
        run_id = int(m.group(5))


        return {
            "Corner Cases": f"{cc}%",
            "Train Size": train_size,
            "Test Unseen": f"{test_unseen}%",
            "LM Model": lm_model,
            "run": run_id,
        }
    else:
        """
    "final_large_20cc80rnd000un_lm=roberta_da=del_dk=None_su=False_size=None_id=0_english"
    "final_large_20cc80rnd000un_lm=roberta_da=del_dk=None_su=False_size=None_id=0_english.txt"
    """

    m = re.search(
        r"final_(small|medium|large)_(\d+)cc\d+rnd(\d+)un_lm=([^_]+).*?_id=(\d+)_english",
        name
    )
    if not m:
        raise ValueError(f"Cannot parse: {name}")

    train_size = m.group(1).capitalize()
    cc = int(m.group(2))
    test_unseen = m.group(3)
    lm_model = m.group(4)
    run_id = int(m.group(5))

    return {
        "Corner Cases": f"{cc}%",
        "Train Size": train_size,
        "Test Unseen": f"{test_unseen}%",
        "LM Model": lm_model,
        "run": run_id,
    }


def compute_metrics(df):
    y_true = df["label"]
    y_pred = df["prediction"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)


    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def ditto_results_analysis():

    # ------------------------------------------------------------------------------
    # Load CodeCarbon JSON logs — THIS IS OUR GROUND-TRUTH FOR GROUPING
    # ------------------------------------------------------------------------------

    cc_records = []
    for jf in glob.glob("data/efficiency_tracker/ditto_en/*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            #use only the last row
            data = data[-1:]

        for rec in data:
            job = rec["job_name"]

            # Example job:
            # learning-curve_ditto_DecisionTree_run1_feature=brand+name+price+desc_train=..._test=...
            meta = parse_dataset_name(job)

            cc_records.append({
                **meta,
                "job_name": job,
                "energy_kwh": rec["energy_kwh"],
                "emissions_kg": rec["emissions_kg"],
                "runtime_sec": rec["runtime_sec"],
                "max_memory_mb": rec["max_memory_mb"],
            })

    carbon_df = pd.DataFrame(cc_records)



    # ------------------------------------------------------------------------------
    # Load performance CSVs (F1, precision, recall, etc.)
    # ------------------------------------------------------------------------------

    perf_rows = []

    for tf in glob.glob("src/models/ditto/output_en/prediction/*.csv", recursive=True):
        name = Path(tf).stem  # ohne .txt
        meta = parse_dataset_name(name, True)


        # to get precision, recall and accuracy we use prediction file
        pred_df = pd.read_csv(tf)
        results = compute_metrics(pred_df)


        perf_rows.append({
            **meta,
            **results,
        })

    perf_df = pd.DataFrame(perf_rows)


    # ------------------------------------------------------------------------------
    # Merge metrics + energy/CO₂
    # ------------------------------------------------------------------------------

    merged = perf_df.merge(
        carbon_df,
        on=["Corner Cases", "Train Size", "Test Unseen", "LM Model", "run"],
        how="left"
    )

    # ------------------------------------------------------------------
    # Keep only experiments that have exactly 3 runs (id 0,1,2)
    # ------------------------------------------------------------------

    group_cols = ["Corner Cases", "Train Size", "Test Unseen", "LM Model"]

    run_counts = (
        merged
        .groupby(group_cols)["run"]
        .nunique()
        .reset_index(name="n_runs")
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
            F1_mean=("f1", "mean"),
            F1_std=("f1", "std"),
            Energy_kWh_mean=("energy_kwh", "mean"),
            CO2_kg_mean=("emissions_kg", "mean"),
            Runtime_mean=("runtime_sec", "mean"),
            Memory_MB_mean=("max_memory_mb", "mean"),
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

    out = Path("src/models/ditto/output_en/analysis/ditto_experiment_summary_en.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print(f"Saved Excel report to {out}")


def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    column_path = "data/processed_en/gold-standards_adjusted"
    predictions_path = "src/models/ditto/output_en/prediction/*"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for file in glob.glob(predictions_path):
        df_predictions = pd.read_csv(file)
        #final_large_20cc80rnd000un_lm=roberta_da=del_dk=None_su=False_id=0_adjusted_testset_predictions_050

        name = file.split('/')[-1]
        m = re.search(
            r"final_(small|medium|large)_(\d+)cc(\d+)rnd(\d+)un_lm=([^_]+).*?_id=(\d+)_english_predictions_(\d+)",
            name
        )
        if not m:
            m = re.search(
                r"final_(small|medium|large)_(\d+)cc(\d+)rnd(\d+)un_lm=([^_]+).*?_id=(\d+)_english_predictions",
                name
            )
            un = m.group(4)
            if not m:
                raise ValueError(f"Cannot parse: {name}")
        else:    
            un = m.group(7)

        train_size = m.group(1)
        cc = m.group(2)
        cc_2 = m.group(3)
        run_id = m.group(6)
        gs_name = f"products{cc}cc{cc_2}rnd{un}un"

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
    os.makedirs("src/models/ditto/output_en/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv("src/models/ditto/output_en/analysis/categories/category_analysis.csv", index=False)

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
    category_summary.to_excel("src/models/ditto/output_en/analysis/categories/category_summary.xlsx", index=False)


if __name__ == "__main__":
    ditto_results_analysis()
    categories()


