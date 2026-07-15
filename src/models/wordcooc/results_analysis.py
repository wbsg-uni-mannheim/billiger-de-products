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
    preprocessed_products20cc80rnd000un_train_large_wordcooc_preprocessed_products20cc80rnd050un_gs
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



def wordcooc_results_analysis():

    # ------------------------------------------------------------------------------
    # Load CodeCarbon JSON logs — THIS IS OUR GROUND-TRUTH FOR GROUPING
    # ------------------------------------------------------------------------------

    cc_records = []
    for jf in glob.glob("data/efficiency_tracker/wordcooc_en/*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Multiple records in {jf}, only using the last one.")
                data = [data[-1]]

        for rec in data:
            job = rec["job_name"]


            # Example job:
            # adjusted_wordcooc_DecisionTree_run1_feature=brand+name+price+desc_train=..._test=...
            m = re.search(
                r"learning-curve_wordcooc_(.*?)_run(\d+)_feature=(.*?)_train=(.*?)_test=(.*)",
                job
            )

            model = m.group(1)
            run = int(m.group(2))
            feature = m.group(3)
            train_name = m.group(4)
            test_name = m.group(5)

            meta = parse_dataset_name(f"{train_name}_{test_name}")

            cc_records.append({
                "model": model,
                "run": run,
                "feature": feature,
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

    for cf in glob.glob("src/models/wordcooc/model_output_en/reports/wordcooc/learning-curve/*.csv", recursive=True):
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
        on=["model", "feature", "run", "train_name", "test_name"],
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

    out = Path("src/models/wordcooc/model_output_en/analysis/wordcooc_experiment_summary_en.xlsx")
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

    for f, sub in df.groupby("file"):
        plt.plot(sub["threshold"], sub["f1"], label=f)

    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs Confidence threshold")
    plt.legend()
    plt.show()




def file_name_transformation(file):
    #src/models/wordcooc/model_output_en/predictions/wordcooc/learning-curve_adjusted/preprocessed_products50cc50rnd000un_train_small_wordcooc_preprocessed_products50cc50rnd000un_train_small_wordcooc_preprocessed_products50cc50rnd100un_gs_LogisticRegression_2.csv

    m = re.match(
        r".*/.*_train_(?P<size>[^_]+).*preprocessed_(?P<dataset>[^_]+)_gs_(?P<model>[^_]+)_(?P<id>\d+)\.csv$",
        file
    )

    if m:
        label = f"{m['size']}_{m['dataset']}_{m['model']}_{m['id']}"
    else:
        raise ValueError(f"Could not parse filename: {file}")
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
        "src/models/wordcooc/model_output_en/analysis/best_thresholds_f1.csv"
    )

    best_total_final = recompute_and_save(
        best_total,
        prediction_glob_pattern,
        "src/models/wordcooc/model_output_en/analysis/best_thresholds_total_f1_ge_0_9.csv"
    )

    return results, best_f1_final, best_total_final

def categories():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'

    predictions_path = "src/models/wordcooc/model_output_en/predictions/wordcooc/learning-curve/*.csv"
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    category_analysis = []
    for file in glob.glob(predictions_path):
        df_predictions = pd.read_csv(file)
        # create new id_left and id_right from pair id, which is id_left#id_right
        df_predictions[["id_left", "id_right"]] = df_predictions["pair_id"].str.split("#", expand=True)
        #turn all ids to int
        df_predictions["id_left"] = df_predictions["id_left"].astype(int)
        df_predictions["id_right"] = df_predictions["id_right"].astype(int)
        name = file.split('/')[-1].removesuffix(".csv")
        rest, model, run_id = name.rsplit("_", 2)
        gs_match = re.search(r"products\d+cc\d+rnd\d+un_gs", name)
        gs_name = gs_match.group()

        train_match = re.search(r"products\d+cc\d+rnd\d+un_train_(small|medium|large)", name)
        train_name = train_match.group()

        print("Processing Testset:", gs_name, " and Trainset:", train_name, " for model:", model, " run:", run_id)
        # Merge for id_left -> shop_cat_left 
        df_merged_left = pd.merge(df_predictions, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
        
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

        cc = re.search(r"\d+cc\d+rnd", gs_name).group()
        un = re.search(r"\d+un", gs_name).group()
        if "small" in train_name:
            size = "small"
        elif "medium" in train_name:
            size = "medium"
        elif "large" in train_name:
            size = "large"
        
        # create new row in df merged that checks wether label == pred_label
        df_merged_both_test["match"] = df_merged_both_test["true_label"] == df_merged_both_test["pred_label"]

        #create csv that goes through all categories that are in left and right and per category get percentage of wrong predictions (match == False) and total count of items in that category
        # Create a csv with columns: category, wrong_percentage, total_count
        
        for cat_pair, count in category_pair_counts.items():

            cat_df = df_merged_both_test[df_merged_both_test["category_pair"] == cat_pair]

            y_true = cat_df["true_label"]
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
    os.makedirs("src/models/wordcooc/model_output_en/analysis/categories", exist_ok=True)
    category_analysis_df.to_csv("src/models/wordcooc/model_output_en/analysis/categories/category_analysis_en.csv", index=False)

    #create an excel file that gets the average of wrong_percentage per category for each group of cc, un, size and model
    category_summary = (
        category_analysis_df
        .groupby(["category_pair", "cc", "un", "size", "model"])
        .agg(
            f1_mean=("f1", "mean"),
            total_count_mean=("total_count", "mean")
        )
        .reset_index()
    )
    category_summary.to_excel("src/models/wordcooc/model_output_en/analysis/categories/category_summary_en.xlsx", index=False)
        
def categories_of_best_models():
    category_path= "src/models/wordcooc/model_output_en/analysis/categories/category_summary_en.xlsx"
    model_results_path = "src/models/wordcooc/model_output_en/analysis/wordcooc_experiment_summary_en.xlsx"

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
    category_analysis.to_excel("src/models/wordcooc/model_output_en/analysis/categories/category_analysis_best_models_en.xlsx", index=False)



if __name__ == "__main__":
    #wordcooc_results_analysis()
    #analyse_confidence_curves(
    #    "src/models/wordcooc/model_output_en/predictions/wordcooc/learning-curve/*.csv"
    #)
    categories()
    categories_of_best_models()

