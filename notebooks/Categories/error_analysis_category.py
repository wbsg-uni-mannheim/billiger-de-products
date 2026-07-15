import pandas as pd

import re

def create_excel():
    def clean_excel_text(x):
        if isinstance(x, str):
            # Entfernt alle illegalen Excel-Zeichen
            return re.sub(r'[\x00-\x1F\x7F]', '', x)
        return x

    # Annahme: Deine Ursprungsdatei heißt 'produkte.xlsx' und hat ein Tabellenblatt 'Produkte'
    CATEGORY_PATH = 'data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name_only_mainentity_with_new_category.pkl.gz'
    df_categories = pd.read_pickle(CATEGORY_PATH, compression='gzip')
    df_categories = df_categories[["name", "desc", "shop_cat", "top_category_mapped" ]]

    # Ziel-Datei, in der die Stichprobe landet
    output_file = 'notebooks/Categories/stichprobe_erroranalyse.xlsx'

    # Anzahl der zufälligen Produkte, die du ziehen willst
    sample_size = 500


    # Zufällige Stichprobe von 500 Zeilen ziehen (ohne Zurücklegen)
    df_sample = df_categories.sample(n=sample_size, random_state=42)
    df_sample = df_sample.applymap(clean_excel_text)

    # Die Stichprobe in eine neue Excel-Datei schreiben
    df_sample.to_excel(output_file, index=False)
    print(f"Eine zufällige Stichprobe von {sample_size} Produkten wurde in '{output_file}' gespeichert.")

def analyze_excel():
    excel_df = pd.read_excel("notebooks/Categories/error_analysis_category.py")
    # [["name", "desc", "shop_cat", "top_category_mapped", "My Category", "Comments" ]]
    