import pandas as pd
from pathlib import Path
import re


def category_distribution():
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    path_list = ["small", "medium", "large"]
    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/blocking_benchmark_final/{path}/")

        # Alle .json.gz Dateien in allen Unterordnern finden
        table_A = pd.read_csv(f"{base_path}/tableA.csv")
        table_B = pd.read_csv(f"{base_path}/tableB.csv")
        

        # Merge for id_left -> shop_cat_left 
        df_merged_A = pd.merge( table_A, df_categories[['id', 'top_category_mapped']], left_on='original_id', right_on='id', how='left', suffixes=('', '_left') ) 
        # Rename shop_cat column correctly 
        df_merged_A.rename(columns={'top_category_mapped': 'shop_cat_left'}, inplace=True) 
        #group by Category and count
        category_counts_A = df_merged_A['shop_cat_left'].value_counts().to_dict()
        
        # Merge for id_right -> shop_cat_right 
        df_merged_B = pd.merge( table_B, df_categories[['id', 'top_category_mapped']], left_on='original_id', right_on='id', how='left', suffixes=('', '_right') ) 
        # Rename shop_cat column from right merge 
        df_merged_B.rename(columns={'top_category_mapped': 'shop_cat_right'}, inplace=True) 
        #group by Category and count
        category_counts_B = df_merged_B['shop_cat_right'].value_counts().to_dict()

        output_dir = Path(f"testing/dataset_analysis_files/blocking/{path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        category_A = pd.DataFrame(
            [
                {
                    "Category": cat,
                    "Total": amount,
                    "Percentage_of_all": round((amount / len(table_A)), 8),
                }
                for cat, amount in category_counts_A.items()
            ]
        )
        category_A.to_csv(output_dir / "category_A.csv", index=False, encoding="utf-8")

        category_B = pd.DataFrame(
            [
                {
                    "Category": cat,
                    "Total": amount,
                    "Percentage_of_all": round((amount / len(table_B)), 8),
                }
                for cat, amount in category_counts_B.items()
            ]
        )
        category_B.to_csv(output_dir / "category_B.csv", index=False, encoding="utf-8")

        for split in ["train", "valid", "test"]:
            train = pd.read_csv(f"{base_path}/{split}.csv")
            df_train = pd.merge(train, df_merged_A[['id', 'original_id', 'shop_cat_left']], left_on='ltable_id', right_on='id', how='left')
            df_train = pd.merge(df_train, df_merged_B[['id', 'original_id', 'shop_cat_right']], left_on='rtable_id', right_on='id', how='left')
            # Count category distribution
            category_counts = (
                df_train[['shop_cat_left', 'shop_cat_right']]
                .stack()
                .value_counts(dropna=False)
                .to_dict()
            )
            total = len(df_train) * 2  # da wir beide Spalten betrachten    
            category_train = pd.DataFrame(
                [
                    {
                        "Category": cat,
                        "Total": amount,
                        "Percentage_of_all": round((amount / total), 8),
                    }
                    for cat, amount in category_counts.items()
                ]
            )
            category_train.to_csv(output_dir / f"category_{split}.csv", index=False, encoding="utf-8")

        

def full_table():

    BASE_DIR = f"testing/dataset_analysis_files/blocking"
    rows = []
    for path in ["small", "medium", "large"]:
        DIR = Path(f"{BASE_DIR}/{path}")

        for csv_file in DIR.glob("category_*.csv"):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                rows.append({
                        "Size": path,
                        "File": csv_file.name,
                        "Category": row["Category"],
                        "Total": row["Total"],
                        "Percentage": round(row["Percentage_of_all"],8),
                    })
            

    # Speichern
    final_df = pd.DataFrame(rows)
    output_path = f"{BASE_DIR}/category_overview_table.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Saved → {output_path}")

def turn_table_to_excel():
    csv_path = Path("testing/dataset_analysis_files/blocking/category_overview_table.csv")
    df = pd.read_csv(csv_path)
    excel_path = Path("testing/dataset_analysis_files/blocking/category_overview_table.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Converted {csv_path} to {excel_path}")
if __name__ == "__main__":
    category_distribution()
    full_table()
    turn_table_to_excel()