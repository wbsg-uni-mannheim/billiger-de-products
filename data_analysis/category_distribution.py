import pandas as pd
from pathlib import Path
import re


def category_distribution_matching():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    path_list = ["gold-standards_adjusted", "training-sets", "validation-sets"]
    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/derived/{path}/")

        # Alle .json.gz Dateien in allen Unterordnern finden
        all_files = list(base_path.glob("*.json.gz"))

        # Optional: Daten sammeln
        for file in all_files:
            if "multi" in file.name:
                continue
            df = pd.read_json(file, lines=True, compression="gzip")
            # Merge for id_left -> shop_cat_left 
            df_merged_left = pd.merge( df, df_categories[['id', 'top_category_mapped']], left_on='id_left', right_on='id', how='left', suffixes=('', '_left') ) 
            # Rename shop_cat column correctly 
            df_merged_left.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
            
            # Merge for id_right -> shop_cat_right 
            df_merged_both = pd.merge( df_merged_left, df_categories[['id', 'top_category_mapped']], left_on='id_right', right_on='id', how='left', suffixes=('', '_right') ) 
            # Rename shop_cat column from right merge 
            df_merged_both.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 

            # Count category distribution
            category_counts = (
                df_merged_both[['shop_cat_left', 'shop_cat_right']]
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
                        "Percentage_of_all": round((amount / total_records), 8),
                    }
                    for cat, amount in category_counts.items()
                ]
            )
            output_df.to_csv(output_file, index=False, encoding="utf-8")
        
def category_distribution_blocking():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    path_list = ["small", "medium", "large"]
    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/blocking_benchmark_final/{path}/")

        # Alle .json.gz Dateien in allen Unterordnern finden
        table_A = pd.read_csv(base_path / "tableA.csv")
        table_B = pd.read_csv(base_path / "tableB.csv")

        df_table_A_merged = pd.merge(table_A, df_categories[['id', 'top_category_mapped']], left_on='original_id', right_on='id', how='left')
        df_table_A_merged.rename(columns={'top_category_mapped': 'shop_cat'}, inplace=True)

        df_table_B_merged = pd.merge(table_B, df_categories[['id', 'top_category_mapped']], left_on='original_id', right_on='id', how='left')
        df_table_B_merged.rename(columns={'top_category_mapped': 'shop_cat'}, inplace=True)

        category_A_counts = df_table_A_merged['shop_cat'].value_counts(dropna=False).to_dict()
        category_B_counts = df_table_B_merged['shop_cat'].value_counts(dropna=False).to_dict()

        #create one dict with both counts and percentage
        total_records = len(df_table_A_merged) + len(df_table_B_merged)
        category_counts = {}
        for cat, amount in category_A_counts.items():
            category_counts[cat] = {
                "Total_A": amount,
                "Total_B": category_B_counts.get(cat, 0),
                "Percentage_of_all": round((amount + category_B_counts.get(cat, 0)) / total_records, 8)
            }
        for cat, amount in category_B_counts.items():
            if cat not in category_counts:
                category_counts[cat] = {
                    "Total_A": 0,
                    "Total_B": amount,
                    "Percentage_of_all": round(amount / total_records, 8)
                }


        output_dir = Path(f"testing/dataset_analysis_files/blocking/{path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{path}_categories.csv"

        output_df = pd.DataFrame(
            [
                {
                    "Category": cat,
                    "Total_A": amount["Total_A"],
                    "Total_B": amount["Total_B"],
                    "Percentage_of_all": round((amount["Total_A"] + amount["Total_B"]) / total_records, 8),
                }
                for cat, amount in category_counts.items()
            ]
        )
        output_df.to_csv(output_file, index=False, encoding="utf-8")
        output_df.to_excel(output_file.with_suffix('.xlsx'), index=False, encoding="utf-8")

def full_table():


    BASE_DIR = Path("notebooks/Categories")

    split_map = {
        "training-sets": "Train",
        "validation-sets": "Valid",
        "gold-standards_adjusted": "Test",
    }

    rows = []

    for folder, split_name in split_map.items():
        folder_path = BASE_DIR / folder
        if not folder_path.exists():
            continue

        for csv_file in folder_path.glob("*.csv"):
            # Filename parsing
            fname = csv_file.stem  # ohne .csv

            # Corner Cases (z.B. 80cc)
            cc_match = re.search(r"(\d+)cc", fname)
            corner_cases = int(cc_match.group(1)) if cc_match else 0

            # Unseen (z.B. 100un)
            un_match = re.search(r"(\d+)un", fname)
            unseen = int(un_match.group(1)) if un_match else 0

            df = pd.read_csv(csv_file, engine="python")
            # Repariere kaputte CSVs mit Komma in Category
            if df.shape[1] > 3:
                # Alles bis zur vorletzten Spalte ist Category
                category_cols = df.columns[:-2]

                df["Category"] = df[category_cols].astype(str).agg(",".join, axis=1)
                df["Total"] = df.iloc[:, -2]
                df["Percentage_of_all"] = df.iloc[:, -1]

                df = df[["Category", "Total", "Percentage_of_all"]]
            for _, row in df.iterrows():
                rows.append({
                    "Train/Valid/Test": split_name,
                    "Corner Cases": corner_cases,
                    "Unseen": unseen,
                    "Category": row["Category"],
                    "Total": row["Total"],
                    "Percentage": round(row["Percentage_of_all"],8),
                })

    final_df = pd.DataFrame(rows)

    # Optional: sortieren
    final_df = final_df.sort_values(
        ["Train/Valid/Test", "Corner Cases", "Unseen", "Category"]
    )

    # Speichern
    output_path = BASE_DIR / "category_overview_table.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Saved → {output_path}")

def turn_table_to_excel():
    csv_path = Path("notebooks/Categories/category_overview_table.csv")
    df = pd.read_csv(csv_path)
    excel_path = Path("notebooks/Categories/category_overview_table.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Converted {csv_path} to {excel_path}")
if __name__ == "__main__":
    #category_distribution_matching()
    #full_table()
    #turn_table_to_excel()
    category_distribution_blocking()