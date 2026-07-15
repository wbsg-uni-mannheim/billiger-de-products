import os
import numpy as np
import torch
import pandas as pd

from transformers import , DataCollatorWithPadding, Trainer, AutoTokenizer
from dataset_en import BaselineClassificationDataset
from sklearn.metrics import f1_score
from modeling import ContrastiveClassifierModel

# =========================
# CONFIG
# =========================
BASE_OUTPUT_DIR = "src/models/r-supCon/reports_en/predictions/"
CHECKPOINT_ROOT = "src/models/r-supCon/reports_en/contrastive-ft-siamese/"
RUNS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PATH = "data/processed_en/gold-standards_adjusted/"
TOKENIZER_NAME = "roberta-base"

unseen = ["000un", "050un", "100un"]
cornercases = ["20cc80rnd", "50cc50rnd", "80cc20rnd"]
sizes = ["small", "medium", "large"]

# =========================
# MAIN LOOP
# =========================
for cc in cornercases:
    for un in unseen:
        for size in sizes:
            #check if the folder for the dataset exists in BASE_OUTPUT_DIR, if yes, skip to the next iteration
            output_name = f"products{cc}{un}-{size}"
            output_dir = os.path.join(BASE_OUTPUT_DIR, output_name)
            if os.path.exists(output_dir):
                print(f"Output for {output_name} already exists, skipping...")
                continue 

            dataset_name = f"products{cc}{un}"
            TEST_FILE = f"{TEST_PATH}/preprocessed_{dataset_name}_gs.pkl.gz"

            if not os.path.exists(TEST_FILE):
                print(f"[WARNING] Missing dataset: {TEST_FILE}")
                continue

            print(f"\n==============================")
            print(f"DATASET: {dataset_name}")
            print(f"==============================")

            
            
            member_probs = []

            # -------------------------
            # Predict per run
            # -------------------------
            for run in range(RUNS):
                dataset_name_dir = f"products{cc}000un"
                run_dir = os.path.join(
                    CHECKPOINT_ROOT,
                    f"{dataset_name_dir}-{size}-all1024-5e-05-0.07-False-roberta-base_adjusted",
                    str(run)
                )

                checkpoints = sorted([
                    d for d in os.listdir(run_dir)
                    if d.startswith("checkpoint-")
                ])

                if len(checkpoints) == 0:
                    print(f"[WARNING] No checkpoints in {run_dir}")
                    continue

                last_checkpoint = checkpoints[-1]
                checkpoint_dir = os.path.join(run_dir, last_checkpoint)
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
                # -------------------------
                # Load dataset
                # -------------------------
            

                model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_dir
                ).to(DEVICE)
                model.resize_token_embeddings(len(tokenizer))
                model.eval()

                test_dataset = BaselineClassificationDataset(
                    TEST_FILE,
                    dataset_type="test",
                    tokenizer=TOKENIZER_NAME,
                    dataset="lspc"
                )

                labels = test_dataset.data["label"].values

                collator = DataCollatorWithPadding(
                    tokenizer=test_dataset.tokenizer,
                    padding="longest",
                    max_length=256
                )


                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator
                )


                outputs = trainer.predict(test_dataset)
                logits = outputs.predictions

                probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
                preds = np.argmax(logits, axis=1)

                member_probs.append(probs)

                # -------------------------
                # Save baseline predictions
                # -------------------------
                output_name = dataset_name + f"-{size}"
                save_dir = os.path.join(BASE_OUTPUT_DIR, output_name, f"run{run}")
                os.makedirs(save_dir, exist_ok=True)

                df = pd.DataFrame({
                    "label": labels,
                    "probability": probs,
                    "prediction": preds
                })

                df.to_csv(os.path.join(save_dir, "baseline_predictions.csv"), index=False)
