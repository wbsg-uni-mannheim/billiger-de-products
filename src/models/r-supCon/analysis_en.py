import pandas as pd
import glob
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os


def parse_dataset_name_rsup_en(name, efficiencytracker):
    if efficiencytracker:
        """
        predict_un000_large_preprocessed_products20cc80rnd000un_gs_id=0_lm=roberta
        """

        m = re.search(
            r"predict_un(\d+)_(small|medium|large)_preprocessed_products(\d+)cc(\d+)rnd(\d+)un_gs_id=(\d+)_lm=([^_]+)",
            name
        )
        if not m:
            raise ValueError(f"Cannot parse: {name}")

        test_unseen = m.group(1)
        train_size = m.group(2).capitalize()
        cc = m.group(3)
        run_id = int(m.group(6))
        lm_model = m.group(7)
        return {
            "Test Unseen": f"{test_unseen}%",
            "Train Size": train_size,
            "LM Model": lm_model,
            "run": run_id,
            "Corner Cases": f"{cc}%",
        }
    else:
        """
        products20cc80rnd000un-large-all1024-5e-05-0.07-False-roberta-base_adjusted
        """

        m = re.search(
            r"^products(\d+)cc(\d+)rnd(\d+)un-(small|medium|large)-all(\d+)-([0-9.e-]+)-([0-9.]+)-([A-Za-z]+)-([A-Za-z0-9_-]+?)(?:_(adjusted))?$",
            name
        )
        if not m:
            raise ValueError(f"Cannot parse: {name}")

        cc =  m.group(1)
        train_size = m.group(4).capitalize()
        lm_model_raw = m.group(9)
        lm_model = lm_model_raw.split("-")[0].split("_")[0]

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
    for jf in glob.glob(f"data/efficiency_tracker/r_supCon_en/*.json"):
        if not jf.__contains__("predict"):
            continue  # only consider roberta-base for now, as we only have those for the contrastive runs
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            #get the last three records (for cc80, cc50 and cc20)
            data = data[-1:]

        for i,rec in enumerate(data):
            job = rec["job_name"]
            meta = parse_dataset_name_rsup_en(job, True)
            cc_records.append({
                **meta,
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

    
    for folder in glob.glob(f"src/models/r-supCon/reports_en/contrastive-ft-siamese/*"):
        if folder.__contains__("roberta-base"):
            for id_folder in glob.glob(f"{folder}/*"):
                name = Path(folder).name
                meta = parse_dataset_name_rsup_en(name, False)
                run_id = Path(id_folder).name
                if not run_id.isdigit():
                    continue  # no info there
                #add run id to meta
                meta["run"] = int(run_id)
                unseens = ["000", "050", "100"]

                results_path = Path(id_folder) / "all_results.json"
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for unseen in unseens:
                        f1 = data.get(f"predict_un{unseen}_f1")
                        recall = data.get(f"predict_un{unseen}_recall")
                        precision = data.get(f"predict_un{unseen}_precision")
                        accuracy = data.get(f"predict_un{unseen}_accuracy")
                        perf_rows.append({
                            **meta,
                            "Test Unseen": f"{unseen}%",
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

    out = Path("src/models/r-supCon/reports_en/analysis/r_supCon_experiment_summary_en.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print(f"Saved Excel report to {out}")


def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    predictions_path = "src/models/r-supCon/reports_en/predictions/*"
    column_path = "data/processed_en/gold-standards_adjusted"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for file in glob.glob(predictions_path):
        df_predictions = pd.read_csv(file)
        if "small" not in file and "medium" not in file and "large" not in file:
            continue
        #test_un000_0_preprocessed_products20cc80rnd000un_gs_large
        df_predictions = pd.read_csv(file)
        print("Processing file:", file)
        name = file.split('/')[-1].removesuffix(".csv")
        m = re.search(r"test_un(\d+)_(\d+)_preprocessed_products(\d+)cc(\d+)rnd(\d+)un_gs_(small|medium|large)", name)
        if not m:
            m = re.search(r"un\d+_un(\d+)_(\d+)_preprocessed_products(\d+)cc(\d+)rnd(\d+)un_gs_(small|medium|large)", name)
            if not m:
                raise ValueError(f"Cannot parse: {name}")
        unseen = m.group(1)
        run_id = int(m.group(2))
        cc = m.group(3)
        cc_2 = m.group(4)
        train_name = m.group(6)

        gs_name = "preprocessed_products" + cc + "cc" + cc_2 + "rnd" + unseen + "un_gs"+".pkl.gz"

        print("Processing Testset:", gs_name, " and Trainset:", train_name, " run:", run_id)
        df_columns = pd.read_pickle(f"{column_path}/{gs_name}", compression="gzip")
        #merge on exact rows for df_column and df_predictions
        df_merged = pd.merge(df_predictions, df_columns[["id_left", "id_right"]], left_index=True, right_index=True, how="left")

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

        if "small" in train_name:
            size = "small"
        elif "medium" in train_name:
            size = "medium"
        elif "large" in train_name:
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
                "un": unseen,
                "size": size,
                "run_id": run_id
            })
    #create datafram for grouped cc, un, size
    category_analysis_df = pd.DataFrame(category_analysis)
    os.makedirs("src/models/r-supCon/reports_en/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv("src/models/r-supCon/reports_en/analysis/categories/category_analysis_en.csv", index=False)

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
    category_summary.to_excel("src/models/r-supCon/reports_en/analysis/categories/category_summary_en.xlsx", index=False)

if __name__ == "__main__":
    transformer_results_analysis_en()
    categories()
