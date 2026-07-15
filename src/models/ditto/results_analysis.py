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

def parse_dataset_name(name):
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
    for jf in glob.glob("data/efficiency_tracker/ditto/*.json"):
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
    pred_path = "src/models/ditto/output_en/prediction/"

    for tf in glob.glob("src/models/ditto/output_en/prediction/*.csv", recursive=True):
        name = Path(tf).stem  # ohne .txt
        meta = parse_dataset_name(name)


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

    # keep only groups with exactly 3 runs
    valid_groups = run_counts[run_counts["n_runs"] == 3]

    # inner-join back to merged to filter
    merged = merged.merge(valid_groups[group_cols], on=group_cols, how="inner")


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

    out = Path("src/models/ditto/output/analysis/ditto_experiment_summary.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print(f"Saved Excel report to {out}")


#TODO Not finished
def category_distribution(file, pairids_wrong_category):
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    
    df = pd.read_json(file, lines=True, compression="gzip")
    # Merge for id_left -> shop_cat_left 
    df_merged_left = pd.merge( df, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
    # Rename shop_cat column correctly 
    df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
    
    # Merge for id_right -> shop_cat_right 
    df_merged_both = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
    # Rename shop_cat column from right merge 
    df_merged_both.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 

    #Keep only pairs in pairids
    df_paired = df_merged_both[df_merged_both['pair_id'].isin(pairids)]

    # Count category distribution
    category_counts_total = (
        df_merged_both[['shop_cat_left', 'shop_cat_right']]
        .stack()
        .value_counts(dropna=False)
        .to_dict()
    )
    category_counts = (
        df_paired[['shop_cat_left', 'shop_cat_right']]
        .stack()
        .value_counts(dropna=False)
        .to_dict()
    )
    total_records = len(df_merged_both)*2  # since we count both left and right categories
    
    output_dir = Path(f"notebooks/Categories/{path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file.name}.csv"

    output_df = pd.DataFrame(
        [
            {
                "Category": cat,
                "Total": amount,
                "Percentage_of_all": round((amount / total_records), 2),
            }
            for cat, amount in category_counts.items()
        ]
    )
    output_df.to_csv(output_file, index=False, encoding="utf-8")

if __name__ == "__main__":
    ditto_results_analysis()


