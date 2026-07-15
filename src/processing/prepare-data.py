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
    attributes = {'name': 50,
                  'brand': 5,
                  'desc': 100}

    for attr, value in attributes.items():
        try:
            row[attr] = ' '.join(row[attr].split(' ')[:value])
        except AttributeError:
            continue
    return row
   
def clean_price(price_input):
    price_input = price_input.fillna('')
    price_input = price_input.replace('nan', '')
    price_input = price_input.str.strip()
    return price_input


if __name__ == '__main__':

    categories = ['products20cc80rnd000un', 'products50cc50rnd000un', 'products80cc20rnd000un']
    train_sizes = ['small', 'medium', 'large']
    valid_types = ['000un', '050un', '100un']

    data = pd.read_pickle('data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name.pkl.gz')

    relevant_cols = ['id', 'product_id', 'brand', 'name', 'desc', 'price']

    for category in categories:
        for valid_type in valid_types:
            out_path = f'data/processed_en/pre-train/{category.replace("000un", valid_type)}/'
            shutil.rmtree(out_path, ignore_errors=True)
            Path(out_path).mkdir(parents=True, exist_ok=True)
            
            for train_size in train_sizes:
                try:
                    ids = pd.read_pickle(f'data/processed_en/training-sets/preprocessed_{category}_train_{train_size}.pkl.gz')
                    ids_valid = pd.read_pickle(f'data/processed_en/validation-sets/preprocessed_{category.replace("000un", valid_type)}_valid_{train_size}.pkl.gz')
                except FileNotFoundError:
                    print('File not found:', f'{category} {valid_type} {train_size}')
                    continue
                relevant_ids = set()
                relevant_ids.update(ids['id_left'])
                relevant_ids.update(ids['id_right'])
                relevant_ids.update(ids_valid['id_left'])
                relevant_ids.update(ids_valid['id_right'])

                data_selection = data[data['id'].isin(relevant_ids)].copy()
                data_selection = data_selection[relevant_cols]
                data_selection = data_selection.reset_index(drop=True)
                data_selection = data_selection.fillna('')
                data_selection['name'] = data_selection['name'].apply(utils.clean_string_2020)
                data_selection['desc'] = data_selection['desc'].apply(utils.clean_string_2020)
                data_selection['brand'] = data_selection['brand'].apply(utils.clean_string_2020)

                #data_selection['price'] = clean_price(data_selection['price'])

                data_selection = data_selection.fillna('')

                data_selection['name'] = data_selection['name'].apply(lambda x: html.unescape(x))
                data_selection['desc'] = data_selection['desc'].apply(lambda x: html.unescape(x))
                data_selection['brand'] = data_selection['brand'].apply(lambda x: html.unescape(x))

                data_selection = data_selection.apply(_cut_lspc_multi, axis=1)

                data_selection = data_selection.replace('', None)

                data_selection.to_pickle(f'{out_path}{category.replace("000un", valid_type)}_train_{train_size}.pkl.gz')