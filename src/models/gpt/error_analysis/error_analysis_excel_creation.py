import pandas as pd
import os

def create_excel_of_wrong_matches(gptmodel, cc, un, specifications,batched=False):
    # Paths
    if batched:
        input_csv = f"src/models/gpt/reports/{gptmodel}/csv_results/products_{cc}_{un}un_batched_english_new_prompt.csv"
    else:
        input_csv = f"src/models/gpt/reports/{gptmodel}/products_{cc}_{un}un.csv"
    output_xlsx = f"src/models/gpt/reports/{gptmodel}/error_analysis/error_analysis_products_{cc}_{un}un_{specifications}.xlsx"

    # Ensure the output directory exists
    output_dir = os.path.dirname(f"src/models/gpt/reports/{gptmodel}/error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV.
    # Changed engine to 'c' to handle NULL bytes.
    df = pd.read_csv(input_csv, encoding="utf-8", engine="c")

    # Keep only rows where Match is 0 or -1
    filtered = df[df["Match"].isin([0, -1])]

    # Write to Excel
    filtered.to_excel(output_xlsx, index=False)

    print(f"Saved {len(filtered)} rows to {output_xlsx}")

if __name__ == "__main__":
    # Example usage
    cc = ["20cc80", "50cc50", "80cc20"]
    un = ["000","050", "100"]
    model = "gpt-5.2"
    specifications = f"english_new_prompt"
    for c in cc:
        for u in un:
            if c == "20cc80":
                create_excel_of_wrong_matches(model, c, u, specifications, batched=True)

