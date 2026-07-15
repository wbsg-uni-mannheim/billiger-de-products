import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import html
from pathlib import Path
import shutil
import utils


def _cut_lspc_multi(row):
    attributes = {
        'name': 50,
        'brand': 5,
        'desc': 100
    }

    for attr, value in attributes.items():
        try:
            row[attr] = ' '.join(str(row[attr]).split(' ')[:value])
        except Exception:
            continue
    return row

def clean_price(price_input):
    price_input = price_input.fillna('')
    price_input = price_input.replace('nan', '')
    price_input = price_input.str.strip()
    return price_input

def normalize_text(df):
    df = df.fillna('')
    for col in ['name', 'desc', 'brand']:
        if col in df.columns:
            df[col] = df[col].apply(utils.clean_string_2020)

    df = df.fillna('')
    for col in ['name', 'desc', 'brand']:
        if col in df.columns:

            df[col] = df[col].apply(html.unescape)

    df = df.apply(_cut_lspc_multi, axis=1)
    

    return df


def extract_single_products(df):
    """
    Handles pairwise datasets by splitting left/right records.
    If dataset already contains single products, returns unchanged.
    """

    if 'name_left' in df.columns:
        left = df[['id_left', 'brand_left', 'name_left', 'desc_left', 'price_left', 'product_id_left']].copy()
        left.columns = ['id', 'brand', 'name', 'desc', 'price', 'product_id']

        right = df[['id_right', 'brand_right', 'name_right', 'desc_right', 'price_right', 'product_id_right']].copy()
        right.columns = ['id', 'brand', 'name', 'desc', 'price', 'product_id']

        combined = pd.concat([left, right]).drop_duplicates(subset='id')
        combined = combined.reset_index(drop=True)
        return combined

    return df


if __name__ == '__main__':

    categories = [
        'products20cc80rnd000un',
        'products50cc50rnd000un',
        'products80cc20rnd000un'
    ]

    train_sizes = ['small', 'medium', 'large']
    valid_types = ['000un', '050un', '100un']

    relevant_cols = ['id', 'product_id', 'brand', 'name', 'desc', 'price']

    for category in categories:
        for valid_type in valid_types:

            out_path = f'data/processed_en/pre-train/{category.replace("000un", valid_type)}/'
            shutil.rmtree(out_path, ignore_errors=True)
            Path(out_path).mkdir(parents=True, exist_ok=True)

            for train_size in train_sizes:
                try:
                    train_df = pd.read_pickle(
                        f'data/processed_en/training-sets/preprocessed_{category}_train_{train_size}.pkl.gz'
                    )

                    valid_df = pd.read_pickle(
                        f'data/processed_en/validation-sets/preprocessed_{category.replace("000un", valid_type)}_valid_{train_size}.pkl.gz'
                    )

                except FileNotFoundError:
                    print('File not found:', f'{category} {valid_type} {train_size}')
                    continue

                # --- Convert pairwise → single products if needed ---
                train_products = extract_single_products(train_df)
                valid_products = extract_single_products(valid_df)

                # --- Combine ---
                data_selection = pd.concat([train_products, valid_products])
                data_selection = data_selection.drop_duplicates(subset='id')

                # --- Keep only relevant columns if present ---
                data_selection = data_selection[relevant_cols]
                data_selection = data_selection.reset_index(drop=True)

                # --- Clean / normalize ---
                data_selection = normalize_text(data_selection)

                # --- Save ---
                output_file = (
                    f'{out_path}'
                    f'{category.replace("000un", valid_type)}_train_{train_size}.pkl.gz'
                )

                data_selection.to_pickle(output_file)

                print("Saved:", output_file)
