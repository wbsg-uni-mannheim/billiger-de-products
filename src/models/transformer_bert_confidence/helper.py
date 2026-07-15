import numpy as np
import pandas as pd
import torch
import os


# ==========================================================
# LOGITS → PROBS + PREDS
# ==========================================================
def logits_to_probs_preds(logits):

    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)

    return probs, preds


# ==========================================================
# SAVE CSV
# ==========================================================
def save_predictions_from_logits(logits_path, dataset_path, output_csv):

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    logits = np.load(logits_path)
    df = pd.read_pickle(dataset_path)

    labels = df["label"].values
    pair_ids = df["pair_id"].astype(str).tolist()

    probs, preds = logits_to_probs_preds(logits)

    out = pd.DataFrame({
        "pair_id": pair_ids,
        "label": labels,
        "probability": probs,
        "prediction": preds
    })

    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    reports_root = "src/models/transformer_bert_confidence/reports/baseline"
    predictions_root = "src/models/transformer_bert_confidence/reports/predictions"

    sizes = ["small", "medium", "large"]
    corner_cases = ["20cc80", "50cc50", "80cc20"]

    unseen_settings = {
        "000un": "",
        "050un": "_un050",
        "100un": "_un100"
    }

    for size in sizes:
        for cc in corner_cases:

            model_base_key = f"products{cc}rnd000un"
            model_dir = f"{reports_root}/{model_base_key}-{size}-all1024-5e-05-roberta-base_adjusted"

            for unseen, suffix in unseen_settings.items():

                dataset_key = f"products{cc}rnd{unseen}"
                dataset_path = (
                    f"data/processed/gold-standards_adjusted/"
                    f"preprocessed_{dataset_key}_gs.pkl.gz"
                )

                pred_dir = f"{predictions_root}/{dataset_key}-{size}"

                for run in range(3):

                    run_dir = f"{model_dir}/{run}"
                    out_dir = f"{pred_dir}/run{run}"

                    save_predictions_from_logits(
                        f"{run_dir}/base_logits{suffix}.npy",
                        dataset_path,
                        f"{out_dir}/baseline_predictions.csv"
                    )
