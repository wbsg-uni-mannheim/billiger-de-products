import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# ==========================================================
# SAVE HELPERS
# ==========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_plot(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ==========================================================
# LOGITS → PROBS + PREDS
# ==========================================================
def logits_to_probs_preds(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1], np.argmax(logits, axis=1)

# ==========================================================
# ECE
# ==========================================================
def compute_ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    return ece

# ==========================================================
# LOAD LOGITS PER RUN
# ==========================================================
def load_outputs_per_run(output_dir, num_runs=3):
    sets = ["", "_un050", "_un100"]
    all_outputs = {}

    for suffix in sets:
        base, temp, mc = [], [], []

        for i in range(num_runs):
            base.append(np.load(f"{output_dir}/{i}/base_logits{suffix}.npy"))
            temp.append(np.load(f"{output_dir}/{i}/temperature_scaled_logits{suffix}.npy"))
            mc.append(np.load(f"{output_dir}/{i}/mc_dropout_logits{suffix}.npy"))

        ens = np.load(f"{output_dir}/ensemble_logits{suffix}.npy")

        all_outputs[suffix] = {
            "base": base,
            "temp": temp,
            "mc": mc,
            "ens": ens
        }

    return all_outputs

# ==========================================================
# EVALUATE MEAN ± STD OVER RUNS
# ==========================================================
def evaluate_across_runs(y_true, logits_list):
    f1s = []
    eces = []

    for logits in logits_list:
        probs, preds = logits_to_probs_preds(logits)
        f1s.append(f1_score(y_true, preds))
        eces.append(compute_ece(probs, y_true))

    return (
        np.mean(f1s), np.std(f1s),
        np.mean(eces), np.std(eces)
    )

# ==========================================================
# BUILD RESULT ROW
# ==========================================================
def build_row(size, unseen, corner_cases, y_true, outputs):
    row = {"Size": size, "Unseen": unseen, "Corner Cases": corner_cases}

    # ----- BASE -----
    f1_m, f1_s, ece_m, ece_s = evaluate_across_runs(y_true, outputs["base"])
    row["F1 Base (%)"] = f"{f1_m*100:.2f} ± {f1_s*100:.2f}"
    row["ECE Base"] = f"{ece_m:.4f} ± {ece_s:.4f}"

    # ----- TEMP -----
    f1_m, f1_s, ece_m, ece_s = evaluate_across_runs(y_true, outputs["temp"])
    row["F1 Temp (%)"] = f"{f1_m*100:.2f} ± {f1_s*100:.2f}"
    row["ECE Temp"] = f"{ece_m:.4f} ± {ece_s:.4f}"

    # ----- MC -----
    f1_m, f1_s, ece_m, ece_s = evaluate_across_runs(y_true, outputs["mc"])
    row["F1 MC (%)"] = f"{f1_m*100:.2f} ± {f1_s*100:.2f}"
    row["ECE MC"] = f"{ece_m:.4f} ± {ece_s:.4f}"

    # ----- ENSEMBLE (single model) -----
    probs, preds = logits_to_probs_preds(outputs["ens"])
    row["F1 Ens (%)"] = f"{f1_score(y_true, preds)*100:.2f}"
    row["ECE Ens"] = f"{compute_ece(probs, y_true):.4f}"

    return row

# ==========================================================
# PRECISION–COVERAGE CURVE
# ==========================================================
def evaluate_threshold(probs, y_true, th):
    preds = (probs >= th).astype(int)
    return {
        "threshold": th,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        "coverage": preds.mean()
    }

def plot_precision_coverage(probs, y_true, title):
    ths = np.linspace(0.5, 0.99, 30)
    prec, cov = [], []

    for t in ths:
        m = evaluate_threshold(probs, y_true, t)
        prec.append(m["precision"])
        cov.append(m["coverage"])

    fig = plt.figure()
    plt.plot(cov, prec)
    plt.xlabel("Coverage")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)

    return fig

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    results_root = "src/models/transformer_bert_confidence/reports_calibration_de"
    all_rows = []

    for size in ["small", "medium", "large"]:
        for corner_cases in ["20cc80", "50cc50", "80cc20"]:

            base_key = f"products{corner_cases}rnd000un"
            output_dir = (
                f"src/models/transformer_bert_confidence/reports/baseline/"
                f"{base_key}-{size}-all1024-5e-05-roberta-base_adjusted"
            )

            df0 = pd.read_pickle(
                f"data/processed/gold-standards_adjusted/preprocessed_{base_key}_gs.pkl.gz"
            )
            df50 = pd.read_pickle(
                f"data/processed/gold-standards_adjusted/preprocessed_{base_key.replace('000un','050un')}_gs.pkl.gz"
            )
            df100 = pd.read_pickle(
                f"data/processed/gold-standards_adjusted/preprocessed_{base_key.replace('000un','100un')}_gs.pkl.gz"
            )

            y_sets = {
                "000un": df0["label"].values,
                "050un": df50["label"].values,
                "100un": df100["label"].values
            }

            outputs = load_outputs_per_run(output_dir, num_runs=3)

            for suffix, unseen_name in zip(["", "_un050", "_un100"], ["000un", "050un", "100un"]):

                print(f"\n=== {corner_cases} | {size} | {unseen_name} ===")

                row = build_row(size, unseen_name, corner_cases, y_sets[unseen_name], outputs[suffix])
                all_rows.append(row)

                save_dir = f"{results_root}/{corner_cases}/{size}/{unseen_name}"
                ensure_dir(save_dir)

                # Precision–Coverage using ensemble
                probs, _ = logits_to_probs_preds(outputs[suffix]["ens"])
                fig = plot_precision_coverage(probs, y_sets[unseen_name], "Precision–Coverage (Ensemble)")
                save_plot(fig, f"{save_dir}/precision_coverage_ensemble.png")

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(f"{results_root}/final_calibration_table.csv", index=False)

    print("\n=== FINAL TABLE ===")
    print(final_df)