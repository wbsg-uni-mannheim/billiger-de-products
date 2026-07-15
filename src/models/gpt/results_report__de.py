import pandas as pd
import glob
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    f1_score
)

# =========================================================
# === CONFIGURATION PATHS
# =========================================================
additional_info = "easy_prompt"  # e.g., "new_prompt", "old_prompt", "other_experiment_info"
GPT_RESULTS_GLOB = "src/models/gpt/reports_de/gpt-5.2/csv_results/*.csv"
CARBON_LOGS_GLOB = "data/efficiency_tracker/gpt_de/gpt-5.2/efficiency_*.json"

OUTPUT_EXCEL = Path(f"src/models/gpt/analysis/gpt_experiment_summary_de_{additional_info}.xlsx")


# =========================================================
# === METRIC COMPUTATION
# =========================================================

def compute_metrics(df):
    y_true = df["Label"]
    y_pred = df["Answer_binary"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )
    f1_ = f1_score(df["Label"], df["Answer_binary"])
    if f1 != f1_:
        print("Warning: F1 mismatch:", f1, f1_)

    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    balanced_acc = (recall + specificity) / 2

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# =========================================================
# === JOB NAME PARSER (FROM CodeCarbon JSON)
# =========================================================

def parse_gpt_job(job_name):
    """
    Example job_name:
    "gpt-5.2_cc20cc80_un050_batched_german_easy_prompt"
    """

    pattern = rf"^gpt-(\d+(?:\.\d+)?)_cc(\d+)cc(\d+)_un(\d+)_batched_german_{re.escape(additional_info)}$"
    m = re.search(pattern, job_name)
    if not m:
        print("Could not parse job name:", job_name)
        return None

    model = m.group(1)
    cc = int(m.group(2))
    unseen = m.group(4)

    return {
        "model": f"gpt-{model}",
        "Test Corner Case": f"{cc}%",
        "Test Unseen": f"{unseen}%"
    }


# =========================================================
# === LOAD CodeCarbon LOGS
# =========================================================

def load_gpt_carbon_logs():
    records = []

    for jf in glob.glob(CARBON_LOGS_GLOB):
        
        if not jf.__contains__(additional_info):
            continue
        if not jf.endswith(".json"):
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[-1:]  # only last run

        for rec in data:
            meta = parse_gpt_job(rec["job_name"])
            if meta is None:
                continue

            records.append({
                **meta,
                "job_name": rec["job_name"],
                "runtime_sec": rec["runtime_sec"],
                "max_memory_mb": rec["max_memory_mb"],
                "energy_kwh": rec["energy_kwh"],
                "emissions_kg": rec["emissions_kg"],
                "gpt_cost_eur": rec["gpt_cost_eur"]
            })

    return pd.DataFrame(records)


# =========================================================
# === LOAD GPT CSV RESULTS + COMPUTE METRICS
# =========================================================

def load_all_gpt_results():
    rows = []

    for cf in glob.glob(GPT_RESULTS_GLOB):
        if not cf.__contains__(additional_info):
            continue
        print("Processing:", cf)

        df = pd.read_csv(cf)

        metrics = compute_metrics(df)

        model = cf.split("/")[-3]

        #products_20cc80_000un_batched_german_easy_prompt
        m = re.search(r"products_(\d+)cc\d+_(\d+)un_[A-Za-z]+", cf)
        if not m:
            print("Could not parse dataset info from:", cf)
            continue

        cc = m.group(1)
        unseen = m.group(2)

        # Hard negative accuracy
        hard_neg_acc = None
        if "Hard_Negative" in df.columns:
            hard_df = df[df["Hard_Negative"] == 1]
            if not hard_df.empty:
                hard_neg_acc = hard_df["Match"].mean()

        rows.append({
            "model": model,
            "Test Corner Case": f"{cc}%",
            "Test Unseen": f"{unseen}%",
            **metrics,
            "hard_negative_accuracy": hard_neg_acc,
            "gpt_cost_eur": df["Costs"].sum()
        })

    return pd.DataFrame(rows)


# =========================================================
# === BUILD SUMMARY
# =========================================================

def build_summary():
    carbon_df = load_gpt_carbon_logs()
    perf_df = load_all_gpt_results()

    merged = perf_df.merge(
        carbon_df,
        on=[
            "model",
            "Test Corner Case",
            "Test Unseen"
        ],
        how="left"
    )

    summary = (
        merged
        .groupby([
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

            Accuracy_mean=("accuracy", "mean"),
            Specificity_mean=("specificity", "mean"),
            FPR_mean=("fpr", "mean"),

            HardNegAcc_mean=("hard_negative_accuracy", "mean"),

            Energy_kWh_mean=("energy_kwh", "mean"),
            CO2_kg_mean=("emissions_kg", "mean"),
            Runtime_mean=("runtime_sec", "mean"),
            Memory_MB_mean=("max_memory_mb", "mean"),
            GPT_Cost_mean=("gpt_cost_eur_x", "mean")
        )
        .reset_index()
    )

    # =====================================================
    # Efficiency metrics
    # =====================================================

    summary["F1_per_kWh"] = summary["Energy_kWh_mean"] / (summary["F1_mean"] + 1e-12)
    summary["F1_per_kgCO2"] = summary["CO2_kg_mean"] / (summary["F1_mean"] + 1e-12)
    summary["F1_per_runtime"] = summary["F1_mean"] / (summary["Runtime_mean"] + 1e-12)
    summary["F1_per_memory"] = summary["F1_mean"] / (summary["Memory_MB_mean"] + 1e-12)
    summary["F1_per_cost"] = summary["F1_mean"] / (summary["GPT_Cost_mean"] + 1e-12)

    cost_cols = [
        "Energy_kWh_mean",
        "CO2_kg_mean",
        "Runtime_mean",
        "Memory_MB_mean",
        "GPT_Cost_mean"
    ]

    for c in cost_cols:
        mn, mx = summary[c].min(), summary[c].max()
        summary[c + "_norm"] = (summary[c] - mn) / (mx - mn + 1e-12)

    summary["Cost_score"] = sum(summary[c + "_norm"] for c in cost_cols)
    summary["Efficiency_Score"] = summary["F1_mean"] / (summary["Cost_score"] + 1e-12)

    return merged, summary


# =========================================================
# === SAVE EXCEL
# =========================================================

def save_excel():
    merged, summary = build_summary()

    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="per_run_data")
        summary.to_excel(writer, index=False, sheet_name="summary_stats")

    print("Saved Excel report →", OUTPUT_EXCEL)


# =========================================================
# === MAIN
# =========================================================

if __name__ == "__main__":
    save_excel()