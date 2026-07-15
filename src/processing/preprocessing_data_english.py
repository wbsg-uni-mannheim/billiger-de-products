import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import glob

import html

from tqdm.auto import tqdm

import utils


def _cut_lspc(row):
    attributes = {'name_left': 50,
                  'name_right': 50,
                  'brand_left': 5,
                  'brand_right': 5,
                  'desc_left': 100,
                  'desc_right': 100}

    for attr, value in attributes.items():
        try:
            row[attr] = ' '.join(row[attr].split(' ')[:value])
        except AttributeError:
            continue
    return row

def _cut_lspc_multi(row):
    attributes = {'name': 50,
                  'brand': 5,
                  'desc': 100,}

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

    # preprocess training sets and gold standards
    print('BUILDING PREPROCESSED TRAINING SETS AND GOLD STANDARDS...')
    os.makedirs(os.path.dirname('data/processed_en/training-sets/'), exist_ok=True)
    os.makedirs(os.path.dirname('data/processed_en/gold-standards_adjusted/'), exist_ok=True)
    os.makedirs(os.path.dirname('data/processed_en/validation-sets/'), exist_ok=True)

    for file in tqdm(glob.glob('data/derived_en/training-sets/*')):
        if 'products' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['name_left'] = df['name_left'].apply(utils.clean_string_2020)
            df['desc_left'] = df['desc_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['name_right'] = df['name_right'].apply(utils.clean_string_2020)
            df['desc_right'] = df['desc_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)
            
            #df['price_left'] = clean_price(df['price_left'])
            #df['price_right'] = clean_price(df['price_right'])
            
            df = df.fillna('')

            df['name_left'] = df['name_left'].apply(lambda x: html.unescape(x))
            df['desc_left'] = df['desc_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['name_right'] = df['name_right'].apply(lambda x: html.unescape(x))
            df['desc_right'] = df['desc_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))
            
            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)
            
            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/training-sets/{file}')

        elif 'products' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['name'] = df['name'].apply(utils.clean_string_2020)
            df['desc'] = df['desc'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)

            #df['price'] = clean_price(df['price'])
            
            df = df.fillna('')

            df['name'] = df['name'].apply(lambda x: html.unescape(x))
            df['desc'] = df['desc'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/training-sets/{file}')

    for file in glob.glob('data/derived_en/validation-sets/*'):
        if 'products' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['name_left'] = df['name_left'].apply(utils.clean_string_2020)
            df['desc_left'] = df['desc_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['name_right'] = df['name_right'].apply(utils.clean_string_2020)
            df['desc_right'] = df['desc_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)

            #df['price_left'] = clean_price(df['price_left'])
            #df['price_right'] = clean_price(df['price_right'])

            df = df.fillna('')

            df['name_left'] = df['name_left'].apply(lambda x: html.unescape(x))
            df['desc_left'] = df['desc_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['name_right'] = df['name_right'].apply(lambda x: html.unescape(x))
            df['desc_right'] = df['desc_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/validation-sets/{file}')

        elif 'products' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['name'] = df['name'].apply(utils.clean_string_2020)
            df['desc'] = df['desc'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)


            #df['price'] = clean_price(df['price'])

            df = df.fillna('')

            df['name'] = df['name'].apply(lambda x: html.unescape(x))
            df['desc'] = df['desc'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/validation-sets/{file}')

    for file in glob.glob('data/derived_en/gold-standards_adjusted/*'):
        if 'products' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['name_left'] = df['name_left'].apply(utils.clean_string_2020)
            df['desc_left'] = df['desc_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['name_right'] = df['name_right'].apply(utils.clean_string_2020)
            df['desc_right'] = df['desc_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)

            #df['price_left'] = clean_price(df['price_left'])
            #df['price_right'] = clean_price(df['price_right'])
            
            df = df.fillna('')

            df['name_left'] = df['name_left'].apply(lambda x: html.unescape(x))
            df['desc_left'] = df['desc_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['name_right'] = df['name_right'].apply(lambda x: html.unescape(x))
            df['desc_right'] = df['desc_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/gold-standards_adjusted/{file}')
        
        elif 'products' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['name'] = df['name'].apply(utils.clean_string_2020)
            df['desc'] = df['desc'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)

            #df['price'] = clean_price(df['price'])

            df = df.fillna('')

            df['name'] = df['name'].apply(lambda x: html.unescape(x))
            df['desc'] = df['desc'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'data/processed_en/gold-standards_adjusted/{file}')

    print('FINISHED BUILDING PREPROCESSED TRAINING SETS AND GOLD STANDARDS...')