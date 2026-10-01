import os
import pandas as pd

def adjust_gold_standard_from_excel(
    excel_path,
    gold_standard_path,
    pair_id_col="Pair_ID",
    pair_id_gs="pair_id",
    label_col="Label",
    new_label_col="New Label",
):
    # Load reviewed Excel
    review_df = pd.read_excel(excel_path, sheet_name="main")
    # make new label column int
    print(f"Loaded {len(review_df)} rows with these columns: {review_df.columns.tolist()}")
    
    #  NaN check in New Label
    nan_rows = review_df[review_df[new_label_col].isna()]
    if not nan_rows.empty:
        row = nan_rows.iloc[0]
        excel_row_number = row.name + 2  # +2 because Excel is 1-based + header
        raise ValueError(
            f"\n❌ NaN detected in '{new_label_col}'\n"
            f"File: {excel_path}\n"
            f"Excel row: {excel_row_number}\n"
            f"{pair_id_col}: {row[pair_id_col]}"
        )
    
    review_df[new_label_col] = review_df[new_label_col].astype(int)
    # Build mapping: Pair_ID -> New Label
    corrections = dict(
        zip(review_df[pair_id_col], review_df[new_label_col])
    )

    # Load gold standard dataset
    for file in os.listdir(gold_standard_path):
        if file.endswith(".json.gz"):
            gold_standard_file = os.path.join(gold_standard_path, file)

            gold_df = pd.read_json(gold_standard_file, compression="gzip", lines=True)

            # Apply corrections
            mask = gold_df[pair_id_gs].isin(corrections)
            gold_df.loc[mask, label_col] = gold_df.loc[mask, pair_id_gs].map(corrections)

            print(f"Amount of to be deleted items: {gold_df[gold_df[label_col] == -1].shape[0]}")

            # Remove where label is -1 (if any)
            gold_df = gold_df[gold_df[label_col] != -1]
            

            # Prepare output path
            os.makedirs(gold_standard_path, exist_ok=True)
            filename = os.path.basename(gold_standard_file)
            output_path = os.path.join(gold_standard_path, filename)

            # Save adjusted dataset
            gold_df.to_json(
                output_path,
                orient="records",
                lines=True,
                compression="gzip",
                force_ascii=False
            )


            print(f"Updated {mask.sum()} rows")
            print(f"Saved adjusted gold standard to {output_path}")


if __name__ == "__main__":
    model = "gpt-5.2"
    # Example usage
    cc = ["20cc80", "50cc50", "80cc20"]
    un = ["000","050", "100"]
    for c in cc:
        for u in un:
                excel_path = f"src/models/gpt/reports_old/{model}/reviewed_new_prompt/error_analysis_products_{c}_{u}un_english_new_prompt_5_2.xlsx"
                gold_path = f"data/derived/gold-standards_adjusted"
                adjust_gold_standard_from_excel(
                excel_path,
                gold_path,
                pair_id_col="Pair_ID",
                pair_id_gs="pair_id",
                label_col="label",      # or whatever your gold label column is named
                new_label_col="New Label",
                )
