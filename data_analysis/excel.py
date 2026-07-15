import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

def dataset_analysis(csv_path, filename):
    
    
    # CSV laden
    df = pd.read_csv(csv_path)
    #remove rows where filename contain other than 000un
    df = df[df["File Name"].str.contains("000un")]

    # Size aus dem Dateinamen extrahieren
    df["Size"] = df["File Name"].str.extract(r"(small|medium|large)")

    # Spalten, über die wir aggregieren wollen (alle numerischen außer File Name)
    value_cols = df.columns.drop(["File Name", "Size"])

    # Gruppieren nach Size und Mean + Std berechnen
    agg = df.groupby("Size")[value_cols].agg(["mean", "std"])

    # Optional: schöner sortieren
    order = ["small", "medium", "large"]
    agg = agg.reindex(order)

    # Als Excel speichern
    agg.to_excel(f"testing/dataset_analysis_files/{filename}.xlsx")

def max_min_percentage_brand(csv_path):
    # CSV laden
    df = pd.read_csv(csv_path)

    # Create percentage of brand and Recordpairs columns
    df["Brand Percentage"] = (df["Amount of brand"] / (df["Recor pairs"]*2))
    
    #return min and max of Brand Percentage
    return df["Brand Percentage"].min(), df["Brand Percentage"].max()

def print_head(validation_set_json_path):
    df = pd.read_json(validation_set_json_path, lines=True, compression="gzip")
    print(df.head())

def compare_df_overlap(df1, df2):
    # Check how many rows are the same in df1 and df2 based on all columns
    #make df1 keep only name_left name_right, brand_left, brand_right, desc_left, desc_right, price_left, price_right
    df1 = df1[['name_left', 'name_right', 'brand_left', 'brand_right', 'desc_left', 'desc_right', 'price_left', 'price_right']]
    df2 = df2[['name_left', 'name_right', 'brand_left', 'brand_right', 'desc_left', 'desc_right', 'price_left', 'price_right']]
    merged = df1.merge(df2, how='inner')
    print("total rows df1:", len(df1))
    print("total rows df2:", len(df2))
    print(f"Number of overlapping rows: {len(merged)}")

def checkunique_words(df1):
    words_df1 = set(
        word
        for col in ['name_left', 'name_right', 'brand_left', 'brand_right', 'desc_left', 'desc_right']
        for text in df1[col].fillna("").str.lower()
        for word in text.split()
    )
    return len(words_df1)

def print_f1(cc, un, gptmodel="gpt-4"):
    df = pd.read_csv(f"src/models/gpt/reports/{gptmodel}/products_{cc}_{un}un_batched.csv")
    f1 = f1_score(df["Label"], df["Answer_binary"])
    print("F1:", f1)

    count_non_match = (df["Match"] == 0).sum()
    print("Wrongly matched by GPT: ", count_non_match)
    
    with open(f"src/models/gpt/reports/{gptmodel}/f1_score_{cc}_{un}un.txt", "w", encoding="utf-8") as f:
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Wrongly matched by GPT: {count_non_match}\n")

    return f1

if __name__ == "__main__":
    #dataset_analysis("testing/dataset_analysis_files/dataset_analysis_training-sets.csv", "dataset_analysis_training-sets")
    #dataset_analysis("testing/dataset_analysis_files/dataset_analysis_validation-sets.csv", "dataset_analysis_validation-sets")

    #print(max_min_percentage_brand("testing/dataset_analysis_files/dataset_analysis_training-sets.csv"))
    #print(max_min_percentage_brand("testing/dataset_analysis_files/dataset_analysis_validation-sets.csv"))

    df1 = pd.read_json("data/derived/validation-sets/products80cc20rnd000un_valid_small.json.gz", lines=True, compression="gzip")
    df2 = pd.read_json("data/derived/validation-sets/products80cc20rnd000un_valid_medium.json.gz", lines=True, compression="gzip")  
    df3 = pd.read_json("data/derived/validation-sets/products80cc20rnd000un_valid_large.json.gz", lines=True, compression="gzip")

    compare_df_overlap(df1, df2)
    compare_df_overlap(df1, df3)
    compare_df_overlap(df2, df3)

    print(checkunique_words(df1))
    print(checkunique_words(df2))
    print(checkunique_words(df3))

    #print(print_f1("80cc20", "050", "gpt-5.2"))

