import pandas as pd
import glob
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

def parse_dataset_name(name):
    """
    Works for strings like:
    preprocessed_products20cc80rnd000un_train_large_magellan_preprocessed_products20cc80rnd050un_gs
    """

    # get both cc + unseen values
    m = re.findall(r'products(\d+)cc\d+rnd(\d+)un', name)

    train_cc = int(m[0][0])
    test_cc = int(m[1][0])

    # unseen in training is always 0 → set explicitly
    train_unseen = 0

    # unseen in test = the LAST rndXXXun
    test_unseen = int(m[-1][1])

    # detect train size
    if "train_small" in name:
        train_size = "Small"
    elif "train_medium" in name:
        train_size = "Medium"
    elif "train_large" in name:
        train_size = "Large"
    else:
        train_size = "Unknown"

    return {
        "Train Corner Cases": f"{train_cc}%",
        "Train Size": train_size,
        "Train Unseen": f"{train_unseen}%",
        "Test Corner Case": f"{test_cc}%",
        "Test Unseen": f"{test_unseen}%"
    }



def magellan_results_analysis():

    # ------------------------------------------------------------------------------
    # Load CodeCarbon JSON logs — THIS IS OUR GROUND-TRUTH FOR GROUPING
    # ------------------------------------------------------------------------------

    cc_records = []
    for jf in glob.glob("data/efficiency_tracker/magellan/*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            #load the last entry of the JSON log (the final metrics after the run is finished)

            data = json.load(f)
            if isinstance(data, list):
                data = [data[-1]]

        for rec in data:
            job = rec["job_name"]
            if "learning-curve_adjusted_testset" in job or "learning-curve_adjusted" not in job:
                continue


            # Example job:
            # learning-curve_magellan_DecisionTree_run1_feature=brand+name+price+desc_train=..._test=...
            print(f"Parsing job: {job}")
            
            m = re.search(
                r"learning-curve_adjusted_([A-Za-z0-9]+)_run(\d+)_testset=([^=]*?\.csv)_trainset=([^=]*?\.csv)",
                job
            )

            model = m.group(1)
            run = int(m.group(2))
            test_name = m.group(3).replace(".csv", "")
            train_name = m.group(4).replace(".csv", "")

            meta = parse_dataset_name(f"{train_name}_{test_name}")

            cc_records.append({
                "model": model,
                "run": run,
                "train_name": train_name,
                "test_name": test_name,

                **meta,

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

    for cf in glob.glob("src/models/magellan/model_output_adjusted_ts/reports/learning-curve_adjusted/*.csv", recursive=True):
        df = pd.read_csv(cf, sep="#####", engine="python")

        perf_rows.append(df)

    perf_df = pd.concat(perf_rows, ignore_index=True)


    # ------------------------------------------------------------------------------
    # Assign runs per model + feature + dataset like before
    # ------------------------------------------------------------------------------

    perf_df["run"] = perf_df.groupby(
        ["model", "feature", "train_set", "test_set"]
    ).cumcount() + 1

    perf_df.rename(columns={
        "train_set": "train_name",
        "test_set": "test_name",
        "precision_test": "precision",
        "recall_test": "recall",
        "f1_test": "f1"
    }, inplace=True)


    # ------------------------------------------------------------------------------
    # Merge metrics + energy/CO₂
    # ------------------------------------------------------------------------------

    merged = perf_df.merge(
        carbon_df,
        on=["model", "run", "train_name", "test_name"],
        how="left"
    )


    # ------------------------------------------------------------------------------
    # Group & compute mean + std (Abweichung)
    # ------------------------------------------------------------------------------

    summary = (
        merged
        .groupby([
            "Train Corner Cases",
            "Train Size",
            "Train Unseen",
            "Test Corner Case",
            "Test Unseen",
            "model"
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

    summary["F1_per_kWh"] = summary["Energy_kWh_mean"] / summary["F1_mean"] # As kWh<1 the lower the better
    summary["F1_per_kgCO2"] = summary["CO2_kg_mean"] / summary["F1_mean"] # As CO2_kg<1 the lower the better
    summary["F1_per_runtime"] = summary["F1_mean"] / summary["Runtime_mean"] # As runtime>1 the lower the better
    summary["F1_per_memory"] = summary["F1_mean"] / summary["Memory_MB_mean"] # As memory>1 the lower the better
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

    out = Path("src/models/magellan/model_output_adjusted_ts/analysis/magellan_experiment_summary_de.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print(f"Saved Excel report to {out}")



def plot_threshold_curve(results, file_filter=None):
    if file_filter:
        df = results[results["file"].str.contains(file_filter)]
    else:
        df = results
    plt.figure(figsize=(10, 6))

    for f, sub in df.groupby("file"):
        plt.plot(sub["threshold"], sub["f1"], label=f)

    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs Confidence threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()

def file_name_transformation(file):
    m = re.match(
    r".*/preprocessed_[^/]*?_train_(?P<size>[^_]+)[^/]*?_formatted_preprocessed_(?P<dataset>[^_]+)_.*_(?P<model>[^_]+)_(?P<id>\d+)\.csv$",
    file
)

    if m:
        label = f"{m['size']}_{m['dataset']}_{m['model']}_{m['id']}"
    return label


def analyse_confidence_curves(prediction_glob_pattern):
    rows = []

    
    for f in glob.glob(prediction_glob_pattern):
        df = pd.read_csv(f)

        y_true = df["true_label"].values
        conf = df["confidence"].values

        if np.isnan(conf).all():
            continue

        thresholds = np.linspace(0, 1, 201)
        dataset_name = file_name_transformation(f)
        total = len(y_true)
        for t in thresholds:
            mask = conf >= t
            if mask.sum() == 0:
                continue

            y_true_sub = y_true[mask]
            y_pred_sub = np.ones_like(y_true_sub)

            tp = (y_true_sub == 1).sum()

            p, r, f1, _ = precision_recall_fscore_support(
                y_true_sub,
                y_pred_sub,
                average="binary",
                zero_division=0
            )

            rows.append({
                "dataset": dataset_name,
                "threshold": t,
                "count": int(mask.sum()),
                "true_positives": int(tp),
                "total": int(total),
                "precision": p,
                "recall": r,
                "f1": f1
            })

    results = pd.DataFrame(rows)

    # global best F1 pro Dataset
    f1_max = results.groupby("dataset")["f1"].max()
 
    def select_best_total(group):
        total = group["total"].iloc[0]

        constrained = group[
            (group["f1"] >= 0.90)
        ]

        if constrained.empty:
            # Fallback: nimm das globale Optimum
            return group.loc[group["f1"].idxmax()]

        # Pick the threshold that keeps the most items
        return constrained.loc[constrained["count"].idxmax()]

    def select_best_f1(group):
        name = group.name
        total = group["total"].iloc[0]  # musst du beim Sammeln speichern
        f1_best = f1_max[name]

        constrained = group[
            (group["f1"] >= 0.90)
        ]

        if constrained.empty:
            # Fallback: nimm das globale Optimum
            return group.loc[group["f1"].idxmax()]

        return constrained.loc[constrained["f1"].idxmax()]

    best_f1 = (
        results
        .groupby("dataset", group_keys=False)
        .apply(select_best_f1)
        .reset_index(drop=True)
    )

    best_total = (
        results
        .groupby("dataset", group_keys=False)
        .apply(select_best_total)
        .reset_index(drop=True)
    )
    def recompute_and_save(best_df, prediction_glob_pattern, out_path):
        final_rows = []

        for f in glob.glob(prediction_glob_pattern):
            name = file_name_transformation(f)
            row = best_df[best_df["dataset"] == name].iloc[0]

            t = row["threshold"]
            df = pd.read_csv(f)

            mask = df["confidence"].values >= t
            y_true_sub = df["true_label"].values[mask]
            y_pred_sub = np.ones_like(y_true_sub)

            p, r, f1, _ = precision_recall_fscore_support(
                y_true_sub,
                y_pred_sub,
                average="binary",
                zero_division=0
            )

            final_rows.append({
                "dataset": name,
                "threshold": t,
                "count": int(mask.sum()),
                "precision": p,
                "recall": r,
                "f1": f1
            })

        out = pd.DataFrame(final_rows)
        out.to_csv(out_path, index=False)
        return out

    best_f1_final = recompute_and_save(
        best_f1,
        prediction_glob_pattern,
        "src/models/magellan/model_output_adjusted_ts/analysis/best_thresholds_f1.csv"
    )

    best_total_final = recompute_and_save(
        best_total,
        prediction_glob_pattern,
        "src/models/magellan/model_output_adjusted_ts/analysis/best_thresholds_total_f1_ge_0_9.csv"
    )

    return results, best_f1_final, best_total_final

def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    column_path = "data/processed/magellan/learning-curve_adjusted"
    predictions_path = "src/models/magellan/model_output_adjusted_ts/predictions/magellan/learning-curve_adjusted/*.csv"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for file in glob.glob(predictions_path):
        df_predictions = pd.read_csv(file)
        name = file.split('/')[-1].removesuffix(".csv")
        rest, model, run_id = name.rsplit("_", 2)
        train_chunk, gs_chunk = rest.split("_formatted_preprocessed_", 1)
        train_name = train_chunk
        train_name = train_name.replace("preprocessed_", "")
        train_name = train_name.replace("_magellan_pairs", "")
        gs_name = "preprocessed_" + gs_chunk.replace("_formatted", "")
        gs_name += ".csv.gz"
        print("Processing Testset:", gs_name, " and Trainset:", train_name, " for model:", model, " run:", run_id)
        df_columns = pd.read_csv(f"{column_path}/{gs_name}", compression="gzip")
        #get pred_label from df_predictions and merge with df_columns to get the category for each pair_id
        df_merged = pd.merge(df_columns, df_predictions[["pair_id", "pred_label"]], left_on='_id', right_on="pair_id", how="left")

        # Merge for id_left -> shop_cat_left 
        df_merged_left = pd.merge(df_merged, df_categories[['id', 'top_category_mapped']], left_on='ltable_id', right_on='id', how='left', suffixes=('', '_left') ) 
        
        # Rename shop_cat column correctly 
        df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
        
        # Merge for id_right -> shop_cat_right 
        df_merged_both_test = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='rtable_id', right_on='id', how='left', suffixes=('', '_right') ) 
        
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

        cc = re.search(r"\d+cc\d+rnd", gs_name).group()
        un = re.search(r"\d+un", gs_name).group()
        if "small" in train_name:
            size = "small"
        elif "medium" in train_name:
            size = "medium"
        elif "large" in train_name:
            size = "large"
        
        # create new row in df merged that checks wether label == pred_label
        df_merged_both_test["match"] = df_merged_both_test["label"] == df_merged_both_test["pred_label"]

        #create csv that goes through all categories that are in left and right and per category get percentage of wrong predictions (match == False) and total count of items in that category
        # Create a csv with columns: category, wrong_percentage, total_count
        
        for cat_pair, count in category_pair_counts.items():

            cat_df = df_merged_both_test[df_merged_both_test["category_pair"] == cat_pair]

            y_true = cat_df["label"]
            y_pred = cat_df["pred_label"]

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
                "model": model,
                "run_id": run_id
            })
    #create datafram for grouped cc, un, size
    category_analysis_df = pd.DataFrame(category_analysis)
    os.makedirs("src/models/magellan/model_output_adjusted_ts/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv("src/models/magellan/model_output_adjusted_ts/analysis/categories/category_analysis_de.csv", index=False)

    #create an excel file that gets the average of f1 per category for each group of cc, un, size and model
    category_summary = (
        category_analysis_df
        .groupby(["category_pair", "cc", "un", "size", "model"])
        .agg(
            f1_mean=("f1", "mean"),
            total_count_mean=("total_count", "mean")
        )
        .reset_index()
    )
    category_summary.to_excel("src/models/magellan/model_output_adjusted_ts/analysis/categories/category_summary_de.xlsx", index=False)
        

def categories_of_best_models():
    category_path= "src/models/magellan/model_output_adjusted_ts/analysis/categories/category_summary_de.xlsx"
    model_results_path = "src/models/magellan/model_output_adjusted_ts/analysis/magellan_experiment_summary_de.xlsx"

    #for each cc, un and size get the best model based on mean f1 score
    model_results = pd.read_excel(model_results_path, sheet_name="summary_stats")
    best_models = (
        model_results
        .groupby(["Train Corner Cases", "Test Unseen", "Train Size"])
        .apply(lambda g: g.loc[g["F1_mean"].idxmax()])
        .reset_index(drop=True)
    )
    #In the Category for each cc, un, size keep only the best model and get the category analysis for it
    category_analysis = pd.read_excel(category_path)
    #make unseen from 000un to 0% and 050un to 50% and 100un to 100% to match the model results
    category_analysis["un"] = category_analysis["un"].str.replace("000un", "0%")
    category_analysis["un"] = category_analysis["un"].str.replace("050un", "50%")
    category_analysis["un"] = category_analysis["un"].str.replace("100un", "100%")

    #make cc from 20cc80rnd to 20% and 50cc50rnd to 50% and 80cc20rnd to 80% to match the model results
    category_analysis["cc"] = category_analysis["cc"].str.replace("20cc80rnd", "20%")
    category_analysis["cc"] = category_analysis["cc"].str.replace("50cc50rnd", "50%")
    category_analysis["cc"] = category_analysis["cc"].str.replace("80cc20rnd", "80%")

    #make size all lowercase in best models to match the category analysis
    best_models["Train Size"] = best_models["Train Size"].str.lower()
    #merge category analysis with best models on cc, un and size
    category_analysis = category_analysis.merge(
        best_models[["Train Corner Cases", "Test Unseen", "Train Size", "model"]],
        left_on=["cc", "un", "size", "model"],
        right_on=["Train Corner Cases", "Test Unseen", "Train Size", "model"],
        how="inner" 
    )
    #save it to a separate xlsx file
    category_analysis.to_excel("src/models/magellan/model_output_adjusted_ts/analysis/categories/category_analysis_best_models_de.xlsx", index=False)
       
    
                

if __name__ == "__main__":
    #magellan_results_analysis()
    #analyse_confidence_curves(
    #    "src/models/magellan/model_output_adjusted_ts/predictions/magellan/learning-curve_adjusted/*.csv"
    #)
    categories()
    categories_of_best_models()

