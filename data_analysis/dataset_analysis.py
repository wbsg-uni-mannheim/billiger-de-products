import pandas as pd
from pathlib import Path
import os
import glob
import gzip
import json
import tiktoken
import regex as re

def create_matching_dataset_analysis():
    
    path_list = ["training-sets", "validation-sets", "gold-standards_adjusted"]

    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/derived_en/{path}/")

        print(base_path)

        # Alle .json.gz Dateien in allen Unterordnern finden
        all_files = list(base_path.glob("*.json.gz"))

        # Optional: Daten sammeln
        rows = []
        for file in all_files:
            if file.name.__contains__("multi"):
                continue
            df = pd.read_json(file, lines=True, compression="gzip")
            rows.append({
                            "File Name": file.name,
                            "Positive Pairs": (df['label'] == 1).sum(),
                            "Negative Pairs": (df['label'] == 0).sum(),
                            "Record pairs": df.shape[0],
                            "Amount of brand": df['brand_left'].notna().sum() + df['brand_right'].notna().sum(),
                            "Amount of name": df['name_left'].notna().sum() + df['name_right'].notna().sum(),
                            "Amount of description": df['desc_left'].notna().sum() + df['desc_right'].notna().sum(),
                            "Amount of price": df['price_left'].notna().sum() + df['price_right'].notna().sum(),
                            "Amount of different products": pd.concat([df['product_id_left'], df['product_id_right']]).nunique(),
                            "Amount of unique Words:": len(
                                set(
                                    word
                                    for col in ['name_left', 'name_right', 'brand_left', 'brand_right', 'desc_left', 'desc_right']
                                    for text in df[col].fillna("").str.lower()
                                    for word in text.split()
                                )
                            )
                        })
        rows_df = pd.DataFrame(rows)
        os.makedirs("testing/dataset_analysis_files/english", exist_ok=True)
        rows_df.to_csv(f"testing/dataset_analysis_files/english/dataset_analysis_{path}.csv", index=False, encoding="utf-8")
        print(f"Daten für {path} gespeichert.")

def amount_of_distind_products():
    path_list = ["training-sets", "validation-sets", "gold-standards_adjusted"]
    ids = set()
    product_ids = set()
    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/derived_en/{path}/")

        print(base_path)

        # Alle .json.gz Dateien in allen Unterordnern finden
        all_files = list(base_path.glob("*.json.gz"))

        print(f"{len(all_files)} Dateien gefunden.")
        for file in all_files:
            if file.name.__contains__("multi"):
                continue
            with gzip.open(file, 'rt', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    ids.add(record['id_left'])
                    ids.add(record['id_right'])
                    product_ids.add(record['product_id_left'])
                    product_ids.add(record['product_id_right'])
    print(f"Anzahl unterschiedlicher IDs: {len(ids)}")
    print(f"Anzahl unterschiedlicher Produkt-IDs: {len(product_ids)}")
        

def create_analysis_excel():
    for file in glob.glob("testing/dataset_analysis_files/english/*.csv"):
        df = pd.read_csv(file)
        excel_file = file.replace(".csv", ".xlsx")
        df.to_excel(excel_file, index=False)
        print(f"Converted {file} to {excel_file}")

def create_blocking_vocabularity():
    TEXT_COLUMNS = ["brand", "name", "desc"]
    path_list = ["large", "medium", "small"]

    for path in path_list:
        base_path = Path(f"data/blocking_benchmark_final/{path}/")

        table_A = pd.read_csv(base_path / "tableA.csv")
        table_B = pd.read_csv(base_path / "tableB.csv")

        combined = pd.concat([table_A, table_B], ignore_index=True)

        total_tokens = 0
        unique_tokens = set()

        # VIEL schneller als iterrows
        for col in TEXT_COLUMNS:
            if col in combined.columns:
                texts = combined[col].dropna().astype(str)

                for text in texts:
                    tokens = text.split()
                    total_tokens += len(tokens)
                    unique_tokens.update(tokens)

        print(f"\nFile: {path}")
        print("Total tokens:", total_tokens)
        print("Unique tokens (Vocabulary size):", len(unique_tokens))

        # Save results
        output_path = Path("testing/dataset_analysis_files/blocking")
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / f"{path}_vocabulary.txt", "w", encoding="utf-8") as f:
            f.write(f"File: {path}\n")
            f.write(f"Total tokens: {total_tokens}\n")
            f.write(f"Unique tokens (Vocabulary size): {len(unique_tokens)}\n\n")
            
    

def create_blocking_dataset_analysis():
    path_list = ["large", "medium", "small"]
    for path in path_list:
        # Pfad zu deinem Hauptordner, in dem die 3 Unterordner liegen
        base_path = Path(f"data/blocking_benchmark_final/{path}/")

        print(base_path)

        # Alle .json.gz Dateien in allen Unterordnern finden
        all_files = list(base_path.glob("*.csv"))

        print(f"{len(all_files)} Dateien gefunden.")

        # Optional: Daten sammeln
        rows = []
        for file in all_files:
            if file.name.__contains__("table"):
                df = pd.read_csv(file)
                rows.append({
                                "File Name": file.name,
                                "Total_products": df.shape[0],
                                "Amount of brand": df['brand'].notna().sum(),
                                "Amount of name": df['name'].notna().sum(),
                                "Amount of description": df['desc'].notna().sum(),
                                "Amount of price": df['price'].notna().sum(),
                                "Amount of different products": df['product_id'].nunique(),
                                "Amount of unique Words:": len(
                                    set(
                                        word
                                        for col in ['name', 'brand', 'desc']
                                        for text in df[col].fillna("").str.lower()
                                        for word in text.split()
                                    )
                                )
                            })
            else:
                df = pd.read_csv(file)
                df_a = pd.read_csv(f"{base_path}/tableA.csv").drop(columns=['product_id'])
                df_b = pd.read_csv(f"{base_path}/tableB.csv").drop(columns=['product_id'])
            
                
                #concat ltable_id of df with id of table a and rtable_id of df with id of table b
                df = (
                    df
                    .merge(df_a, left_on='ltable_id', right_on='id', how='left', suffixes=('', '_left'))
                    .merge(df_b, left_on='rtable_id', right_on='id', how='left', suffixes=('', '_right'))
                )
                print("Columns of df:", df.columns)
                rows.append({
                                "File Name": file.name,
                                "Positive Pairs": (df['label'] == 1).sum(),
                                "Negative Pairs": (df['label'] == 0).sum(),
                                "Record pairs": df.shape[0],
                                "Amount of brand": df['brand'].notna().sum() + df['brand_right'].notna().sum(),
                                "Amount of name": df['name'].notna().sum() + df['name_right'].notna().sum(),
                                "Amount of description": df['desc'].notna().sum() + df['desc_right'].notna().sum(),
                                "Amount of price": df['price'].notna().sum() + df['price_right'].notna().sum(),
                                "Amount of different products": pd.concat(
                                    [df['product_id_left'], df['product_id_right']],
                                    ignore_index=True
                                ).nunique(),
                                "Amount of unique Words:": len(
                                    set(
                                        word
                                        for col in ['name', 'name_right', 'brand', 'brand_right', 'desc', 'desc_right']
                                        for text in df[col].fillna("").str.lower()
                                        for word in text.split()
                                    )
                                )
                            })


        rows_df = pd.DataFrame(rows)
        #check if path exists

        if not os.path.exists("testing/dataset_analysis_files/blocking"):
            os.makedirs("testing/dataset_analysis_files/blocking")
        rows_df.to_csv(f"testing/dataset_analysis_files/blocking/dataset_analysis_{path}.csv", index=False, encoding="utf-8")
        print(f"Daten für {path} gespeichert.")

def create_word_specific_analysis_train():
    COLUMNS = ["brand", "name", "desc", "price"]
    # Choose encoding (depends on model family)
    encoding = tiktoken.get_encoding("cl100k_base")

    def word_count(text, col):
        if col == "price":
            return 1 if text else 0
        return len(re.findall(r"\b\w+\b", text))
    
    def token_count(text):
        return len(encoding.encode(text))

    def density(products, col):
        count = 0
        for product in products.values():
            text = product.get(col)
            if text is None or text == "":
                continue
            count += 1
        return int(count/len(products)*100)

    def median_length(products, col):
        lengths = []
        for product in products.values():
            text = product.get(col) or ""
            if text:
                lengths.append(word_count(text, col))
        if lengths:
            return int(pd.Series(lengths).median())
        return 0

    def density_and_median(products, col):
        density_ = density(products, col)
        median_len = median_length(products, col)
        return f"{density_}/{median_len}"

    def unique_words(products):
        set_of_words = set()   
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    words = set(text.lower().split())
                    set_of_words.update(words)
        return len(set_of_words)

    def unique_tokens(products):
        set_of_tokens = set()
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    tokens = set(encoding.encode(text))
                    set_of_tokens.update(tokens)

        return len(set_of_tokens)

    def collect_unique_products(files):
        products = {}
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as infile:
                for line in infile:
                    r = json.loads(line)

                    for side in ("left", "right"):
                        id = str(r[f"id_{side}"])
                        if id not in products:
                            products[id] = {
                                "id": id,
                                "brand": str(r.get(f"brand_{side}")),
                                "name": str(r.get(f"name_{side}")),
                                "desc": str(r.get(f"desc_{side}")),
                                "price": str(r.get(f"price_{side}")),
                            }
        return products

    def vocabulary_size(products):
        all_words = set()
        for product in products.values():
            text = " ".join([str(product.get(col, "")) for col in COLUMNS if col != "price"])
            words = set(text.lower().split())
            all_words.update(words)
        return len(all_words)

    path_list = ["training-sets", "validation-sets"]
    corner_cases= ["20cc80rnd", "50cc50rnd", "80cc20rnd"]
    sizes = ["small", "medium", "large"]
    combination_files = {}
    rows = {}
    
    
    # get all groups of combinataions of corner cases and size based on their appearance in the file name
   
    base_path = Path(f"data/derived/{path_list[0]}/")
    all_files = list(base_path.glob("*.json.gz"))
    for corner_case in corner_cases:
        for size in sizes:
            combination = f"{corner_case}_{size}"
            #append to combination files if key exists otherwise create new key
            if combination in combination_files:
                combination_files[combination].extend([file for file in all_files if corner_case in file.name and size in file.name and "multi" not in file.name])
            else:
                combination_files[combination] = [file for file in all_files if corner_case in file.name and size in file.name and "multi" not in file.name]

    rows = []
    for combination, files in combination_files.items():
        products = collect_unique_products(files)
        print(f"Combination: {combination}, Number of unique products: {len(products)}")

        row = {
            "Combination": combination,
            "# Entities": len(products),

            # Density / Median Length
            "name": density_and_median(products, "name"),
            "description": density_and_median(products, "desc"),
            "brand": density_and_median(products, "brand"),
            "price": density_and_median(products, "price"),

            "Vocabulary (unique)": vocabulary_size(products),
            "Words": unique_words(products),
            "Tokens": unique_tokens(products)  # approximation
        }

        rows.append(row)

    rows_df = pd.DataFrame(rows)
    os.makedirs("testing/dataset_analysis_vocab/german", exist_ok=True)
    rows_df.to_csv(
        f"testing/dataset_analysis_vocab/german/attribute_infos_{path_list[0]}.csv",
        index=False,
        encoding="utf-8"
    )

    print(f"Daten gespeichert.")

def create_word_specific_analysis_test():
    COLUMNS = ["brand", "name", "desc", "price"]
    # Choose encoding (depends on model family)
    encoding = tiktoken.get_encoding("cl100k_base")

    def word_count(text, col):
        if col == "price":
            return 1 if text else 0
        return len(re.findall(r"\b\w+\b", text))
    
    def token_count(text):
        return len(encoding.encode(text))

    def density(products, col):
        count = 0
        for product in products.values():
            text = product.get(col)
            if text is None or text == "":
                continue
            count += 1
        return int(count/len(products)*100)

    def median_length(products, col):
        lengths = []
        for product in products.values():
            text = product.get(col) or ""
            if text:
                lengths.append(word_count(text, col))
        if lengths:
            return int(pd.Series(lengths).median())
        return 0

    def density_and_median(products, col):
        density_ = density(products, col)
        median_len = median_length(products, col)
        return f"{density_}/{median_len}"

    def unique_words(products):
        set_of_words = set()   
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    words = set(text.lower().split())
                    set_of_words.update(words)
        return len(set_of_words)

    def unique_tokens(products):
        set_of_tokens = set()
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    tokens = set(encoding.encode(text))
                    set_of_tokens.update(tokens)

        return len(set_of_tokens)

    def collect_unique_products(files):
        products = {}
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as infile:
                for line in infile:
                    r = json.loads(line)

                    for side in ("left", "right"):
                        id = str(r[f"id_{side}"])
                        if id not in products:
                            products[id] = {
                                "id": id,
                                "brand": str(r.get(f"brand_{side}")),
                                "name": str(r.get(f"name_{side}")),
                                "desc": str(r.get(f"desc_{side}")),
                                "price": str(r.get(f"price_{side}")),
                            }
        return products

    def vocabulary_size(products):
        all_words = set()
        for product in products.values():
            text = " ".join([str(product.get(col, "")) for col in COLUMNS if col != "price"])
            words = set(text.lower().split())
            all_words.update(words)
        return len(all_words)

    path_list = ["gold-standards_adjusted"]
    corner_cases= ["20cc80rnd", "50cc50rnd", "80cc20rnd"]
    seen = ["000", "050", "100"]
    combination_files = {}
    rows = {}
    
    
    # get all groups of combinataions of corner cases and seen values based on their appearance in the file name
   
    base_path = Path(f"data/derived/{path_list[0]}/")
    all_files = list(base_path.glob("*.json.gz"))
    for corner_case in corner_cases:
        for seen_value in seen:
            combination = f"{corner_case}_{seen_value}"
            #append to combination files if key exists otherwise create new key
            if combination in combination_files:
                combination_files[combination].extend([file for file in all_files if corner_case in file.name and seen_value in file.name and "multi" not in file.name])
            else:
                combination_files[combination] = [file for file in all_files if corner_case in file.name and seen_value in file.name and "multi" not in file.name]

    rows = []
    for combination, files in combination_files.items():
        products = collect_unique_products(files)
        print(f"Combination: {combination}, Number of unique products: {len(products)}")

        row = {
            "Combination": combination,
            "# Entities": len(products),

            # Density / Median Length
            "name": density_and_median(products, "name"),
            "description": density_and_median(products, "desc"),
            "brand": density_and_median(products, "brand"),
            "price": density_and_median(products, "price"),

            "Vocabulary (unique)": vocabulary_size(products),
            "Words": unique_words(products),
            "Tokens": unique_tokens(products)  # approximation
        }

        rows.append(row)

    rows_df = pd.DataFrame(rows)
    os.makedirs("testing/dataset_analysis_vocab/german", exist_ok=True)
    rows_df.to_csv(
        f"testing/dataset_analysis_vocab/german/attribute_infos_{path_list[0]}.csv",
        index=False,
        encoding="utf-8"
    )

    print(f"Daten gespeichert.")
    
def create_overlap_analysis_with_gold_average():
    import json, gzip, re, os
    import pandas as pd
    from pathlib import Path
    import tiktoken
    import numpy as np

    COLUMNS = ["brand", "name", "desc", "price"]
    encoding = tiktoken.get_encoding("cl100k_base")

    corner_cases = ["20cc80rnd", "50cc50rnd", "80cc20rnd"]
    sizes = ["small", "medium", "large"]
    seen_values = ["000", "050", "100"]

    def collect_unique_products(files):
        products = {}
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as infile:
                for line in infile:
                    r = json.loads(line)

                    for side in ("left", "right"):
                        id = str(r[f"id_{side}"])
                        if id not in products:
                            products[id] = {
                                "id": id,
                                "brand": str(r.get(f"brand_{side}")),
                                "name": str(r.get(f"name_{side}")),
                                "desc": str(r.get(f"desc_{side}")),
                                "price": str(r.get(f"price_{side}")),
                            }
        return products

    def extract_word_set(products):
        words = set()
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    found = re.findall(r"\b\w+\b", text.lower())
                    words.update(found)
        return words

    def extract_token_set(products):
        tokens = set()
        for product in products.values():
            for col in COLUMNS:
                if col == "price":
                    continue
                text = product.get(col) or ""
                if text:
                    encoded = encoding.encode(text)
                    tokens.update(encoded)
        return tokens

    train_base = Path("data/derived/training-sets/")
    gold_base = Path("data/derived/gold-standards_adjusted/")

    train_files = list(train_base.glob("*.json.gz"))
    gold_files = list(gold_base.glob("*.json.gz"))

    rows = []

    for corner_case in corner_cases:

        # ---- TRAIN GROUPS
        train_vocab = {}
        for size in sizes:
            key = f"{corner_case}_{size}"
            files = [
                f for f in train_files
                if corner_case in f.name and size in f.name and "multi" not in f.name
            ]
            products = collect_unique_products(files)
            train_vocab[key] = {
                "words": extract_word_set(products),
                "tokens": extract_token_set(products)
            }

        # ---- GOLD GROUPS
        gold_vocab = {}
        for seen in seen_values:
            key = f"{corner_case}_{seen}"
            files = [
                f for f in gold_files
                if corner_case in f.name and seen in f.name and "multi" not in f.name
            ]
            products = collect_unique_products(files)
            gold_vocab[key] = {
                "words": extract_word_set(products),
                "tokens": extract_token_set(products)
            }

        # ---- COMPARISONS
        for train_key, train_data in train_vocab.items():

            per_seen_metrics = []

            for gold_key, gold_data in gold_vocab.items():

                shared_words = train_data["words"] & gold_data["words"]
                shared_tokens = train_data["tokens"] & gold_data["tokens"]

                unique_test_words = gold_data["words"] - train_data["words"]
                unique_test_tokens = gold_data["tokens"] - train_data["tokens"]

                row = {
                    "Corner Case": corner_case,
                    "Train Group": train_key,
                    "Gold Group": gold_key,

                    "Train Words": len(train_data["words"]),
                    "Gold Words": len(gold_data["words"]),
                    "Shared Words": len(shared_words),
                    "Test-only Words": len(unique_test_words),

                    "Train Tokens": len(train_data["tokens"]),
                    "Gold Tokens": len(gold_data["tokens"]),
                    "Shared Tokens": len(shared_tokens),
                    "Test-only Tokens": len(unique_test_tokens),
                }

                rows.append(row)
                per_seen_metrics.append(row)

            # ---- AVERAGE ACROSS SEEN
            avg_row = {
                "Corner Case": corner_case,
                "Train Group": train_key,
                "Gold Group": "AVERAGE"
            }

            for metric in [
                "Gold Words", "Shared Words", "Test-only Words",
                "Gold Tokens", "Shared Tokens", "Test-only Tokens"
            ]:
                avg_row[metric] = int(np.mean([r[metric] for r in per_seen_metrics]))

            avg_row["Train Words"] = per_seen_metrics[0]["Train Words"]
            avg_row["Train Tokens"] = per_seen_metrics[0]["Train Tokens"]

            rows.append(avg_row)

    df = pd.DataFrame(rows)

    os.makedirs("testing/dataset_analysis_vocab/german", exist_ok=True)
    df.to_csv(
        "testing/dataset_analysis_vocab/german/train_vs_gold_overlap_with_avg.csv",
        index=False,
        encoding="utf-8"
    )

    print("Detailed overlap + averages saved.")



if __name__ == "__main__":
    #create_matching_dataset_analysis()
    create_blocking_dataset_analysis()
    #create_analysis_excel()
    #amount_of_distind_products()
    #create_word_specific_analysis_test()
    #create_word_specific_analysis_train()
    #create_overlap_analysis_with_gold_average()
    create_blocking_vocabularity()