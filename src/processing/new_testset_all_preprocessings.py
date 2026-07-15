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

def preprocess_data():

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


    # preprocess adjusted gold standards
    print('BUILDING PREPROCESSED ADJUSTED GOLD STANDARDS...')
    os.makedirs('data/processed/gold-standards_adjusted/', exist_ok=True)

    for file in glob.glob('data/derived/gold-standards_adjusted/*'):
        print(f'Processing file: {file}')
        if 'products' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True, compression='gzip')

            df['name_left'] = df['name_left'].apply(utils.clean_string_2020)
            df['desc_left'] = df['desc_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['name_right'] = df['name_right'].apply(utils.clean_string_2020)
            df['desc_right'] = df['desc_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)

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
            file = f'preprocessed_{file}'
            file = os.path.basename(file).replace('.json.gz', '.pkl.gz')
            df.to_pickle(f'data/processed/gold-standards_adjusted/{file}')

    print('FINISHED BUILDING PREPROCESSED ADJUSTED GOLD STANDARDS...')

def preprocess_data_for_ditto_hiergat():
    import json
    import numpy as np
    import pandas as pd

    import os
    from pathlib import Path
    import glob
    import gzip
    from copy import deepcopy

    import argparse
    import re
    import csv

    def combine_row(left, right, label):
        def func(row):
            col_names = left.columns # assume left and right always have same attributes
            list_ = ['COL' + ' ' + str(b) + ' ' + 'VAL' + ' ' + str(a) + ' ' for a, b in zip(row, col_names.values.tolist())]
            list_ = ''.join(str(m) for m in list_)
            return list_

        left_list = list(map(func, left.values.tolist()))
        right_list = list(map(func, right.values.tolist()))
        label_list = [str(l) for l in label]

        left_df = pd.DataFrame({'left': pd.Series(left_list)})
        right_df = pd.DataFrame({'right': pd.Series(right_list)})
        label_df = pd.DataFrame({'label': pd.Series(label_list)})

        # using tab separator here
        # https://github.com/megagonlabs/ditto - <entry_1> \t <entry_2> \t <label>
        final_df = left_df.left.map(str) + '\t' + right_df.right
        final_df = final_df.map(str) + '\t' + label_df.label

        return final_df

    def preprocess_part(path):
        with gzip.open(f'{path}', 'rb') as f:
            test_set = pd.read_pickle(f.name)

        test_set = test_set.drop(['pair_id'], axis=1)

        mask_left = test_set.columns.str.endswith('_left')
        mask_right = test_set.columns.str.endswith('_right')

        left = test_set.loc[:, mask_left]
        right = test_set.loc[:, mask_right]
        label = [int(x) for x in list(test_set['label'].values)]

        left.columns = left.columns.str.removesuffix('_left')
        left = left.drop(['id', 'product_id'], axis=1)
        left = left.fillna('')
        left = left[['brand', 'name', 'price', 'desc']]

        right.columns = right.columns.str.removesuffix('_right')
        right = right.drop(['id', 'product_id'], axis=1)
        right = right.fillna('')
        right = right[['brand', 'name', 'price', 'desc']]

        final_df = combine_row(left, right, label)

        return(final_df)

    def preprocess_dataset():
        os.makedirs('data/processed/ditto/data/final_output/', exist_ok=True)
        os.makedirs('data/processed/hiergat/data/final_output/', exist_ok=True)

        test_path =  "data/processed/gold-standards_adjusted"

        if not (os.path.exists(test_path)):
            print('Dataset does not exist', test_path)
            return

        print(f'START BULDING FINAL DATASETS')

        for path in glob.glob(os.path.join(test_path, (r'preprocessed_products[!multi]*.pkl.gz'))):
            filename = os.path.basename(path).replace('.pkl.gz', '_adjusted')
            print(f'Processing file: {filename}')
            test_df = preprocess_part(path)
            np.savetxt(f'data/processed/ditto/data/final_output/{filename}.txt', test_df.values, fmt = "%s")
            np.savetxt(f'data/processed/hiergat/data/final_output/{filename}.txt', test_df.values, fmt = "%s")
        print(f'DONE TEST')
        print(f'FINISHED BULDING FINAL DATASETS\n')
    
    preprocess_dataset()

def process_to_magellan():
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)

    import os
    import glob
    import py_entitymatching as em
    from pathlib import Path

    def preprocess_magellan(file, columns_to_preprocess, experiment_name, validation_set=None):
        columns_preprocess_magellan = ['ltable_' + col for col in columns_to_preprocess]
        columns_preprocess_magellan.extend(['rtable_' + col for col in columns_to_preprocess])
        data_df = None
        if '.pkl.gz' in file:
            data_df = pd.read_pickle(file, compression='gzip')
            if 'products' in file and 'gs' not in file:
                val_df = pd.read_pickle(validation_set, compression='gzip')
                data_df = pd.concat([data_df, val_df])
            data_df = data_df.reset_index(drop=True)
        elif '.json.gz' in file:
            data_df = pd.read_json(file, lines=True, compression='gzip')
        else:
            print(f'unrecognized file format: {Path(file).suffix}')
        data_df.fillna('', inplace=True)
        if 'price' in columns_to_preprocess and 'products' not in file:
            data_df['price_left'] = data_df['price_left'].replace(r'^\s*$', np.nan, regex=True)
            data_df['price_right'] = data_df['price_right'].replace(r'^\s*$', np.nan, regex=True)
            data_df['price_left'] = data_df['price_left'].astype('float64')
            data_df['price_right'] = data_df['price_right'].astype('float64')
        # change column naming to magellan format
        cols = list(data_df.columns)
        for i, col in enumerate(cols):
            if '_left' in col:
                col = col.replace('_left', '')
                cols[i] = 'ltable_' + col
            if '_right' in col:
                col = col.replace('_right', '')
                cols[i] = 'rtable_' + col
        data_df.columns = cols

        # build left and right subsets
        left_df = data_df[[col for col in data_df.columns if 'ltable_' in col]].copy()
        left_df.drop_duplicates(subset='ltable_id', inplace=True)
        right_df = data_df[[col for col in data_df.columns if 'rtable_' in col]].copy()
        right_df.drop_duplicates(subset='rtable_id', inplace=True)

        # assign magellan ids in subsets
        left_df['mag_id'] = range(0, len(left_df))
        right_df['mag_id'] = range(0, len(right_df))

        # use magellan ids and assign global pair id
        len_assert = len(data_df)
        data_df = data_df.merge(left_df[['ltable_id', 'mag_id']], how='left', on='ltable_id')
        data_df.rename(columns={'mag_id': 'ltable_mag_id'}, inplace=True)
        data_df = data_df.merge(right_df[['rtable_id', 'mag_id']], how='left', on='rtable_id')
        data_df.rename(columns={'mag_id': 'rtable_mag_id'}, inplace=True)
        data_df['_id'] = range(0, len(data_df))
        assert len(data_df) == len_assert

        left_df.drop(columns='ltable_id', inplace=True)
        right_df.drop(columns='rtable_id', inplace=True)

        left_cols = left_df.columns
        left_df.columns = [col.replace('ltable_', '') for col in left_cols]

        right_cols = right_df.columns
        right_df.columns = [col.replace('rtable_', '') for col in right_cols]

        file_name = os.path.basename(file)
        new_file_name = file_name.replace('.pkl.gz', '_magellan_')
        new_file_name = new_file_name.replace('.json.gz', '_magellan_')

        out_path1 = f'data/processed/magellan/{experiment_name}/'
        out_path2 = f'data/processed/magellan/{experiment_name}/formatted/'

        os.makedirs(out_path2, exist_ok=True)

        left_df.to_csv(out_path1 + new_file_name + 'left.csv.gz', compression='gzip', header=True, index=False)
        right_df.to_csv(out_path1 + new_file_name + 'right.csv.gz', compression='gzip', header=True, index=False)
        data_df.to_csv(out_path1 + new_file_name + 'pairs.csv.gz', compression='gzip', header=True, index=False)

        # magellan formatting for py_entitymatching
        A = em.read_csv_metadata(out_path1 + new_file_name + 'left.csv.gz', key='mag_id')
        em.to_csv_metadata(A, out_path2 + new_file_name + 'left_formatted.csv')
        B = em.read_csv_metadata(out_path1 + new_file_name + 'right.csv.gz', key='mag_id')
        em.to_csv_metadata(B, out_path2 + new_file_name + 'right_formatted.csv')

        C = em.read_csv_metadata(out_path1 + new_file_name + 'pairs.csv.gz',
                                key='_id',
                                ltable=A, rtable=B,
                                fk_ltable='ltable_mag_id', fk_rtable='rtable_mag_id')

        if isinstance(validation_set, type(None)):

            em.to_csv_metadata(C, out_path2 + new_file_name + 'pairs_formatted.csv')

        else:
            if 'csv' in validation_set:
                validation_ids_df = pd.read_csv(validation_set)
                
            else:
                validation_ids_df = pd.read_pickle(validation_set, compression='gzip')

            validation_df = C[C['pair_id'].isin(validation_ids_df['pair_id'].values)]
            train_df = C[~C['pair_id'].isin(validation_ids_df['pair_id'].values)]

            em.to_csv_metadata(C, out_path2 + new_file_name + 'pairs_formatted.csv')

            new_file_name = new_file_name.replace('train', 'trainonly')

            em.to_csv_metadata(train_df, out_path2 + new_file_name + 'pairs_formatted.csv')

            valid_name = new_file_name.replace('trainonly', 'valid')

            em.to_csv_metadata(validation_df, out_path2 + valid_name + 'pairs_formatted.csv')


    for file in glob.glob('data/processed/gold-standards_adjusted/*'):
        columns_to_preprocess = ['name', 'desc', 'brand', 'price']

        preprocess_magellan(file, columns_to_preprocess, experiment_name='learning-curve_adjusted', validation_set=None)

def process_to_wordcooc():
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)

    import os
    import glob
    import json
    from pathlib import Path

    from sklearn.feature_extraction.text import CountVectorizer

    from pdb import set_trace

    def process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations):
        data_df = None
        if '.pkl.gz' in file:
            data_df = pd.read_pickle(file)
        if 'products' in file and 'training' in file:
            valid = file.replace('training', 'validation')
            valid = valid.replace('train', 'valid')
            valid_df = pd.read_pickle(valid)
            data_df = pd.concat([data_df, valid_df])
            data_df = data_df.reset_index(drop=True)
        elif '.json.gz' in file:
            data_df = pd.read_json(file, lines=True, compression='gzip')
        else:
            print(f'unrecognized file format: {Path(file).suffix}')
        data_df.fillna('', inplace=True)# type: ignore

        # preprocess selected columns
        for column in columns_preprocess_wordcooc:
            data_df[column] = data_df[column].astype(str)# type: ignore

        # build combined features for every feature combination
        for feature_combination in feature_combinations:
            feats_to_combine = feature_combination.split('+')
            data_df[feature_combination + '_wordocc_left'] = data_df[feats_to_combine[0] + '_left']# type: ignore
            data_df[feature_combination + '_wordocc_right'] = data_df[feats_to_combine[0] + '_right']# type: ignore

            for feat_to_combine in feats_to_combine[1:]:
                data_df[feature_combination + '_wordocc_left'] += (' ' + data_df[feat_to_combine + '_left'])# type: ignore
                data_df[feature_combination + '_wordocc_right'] += (' ' + data_df[feat_to_combine + '_right'])# type: ignore

            data_df[feature_combination + '_wordocc_left'] = data_df[feature_combination + '_wordocc_left'].str.strip()# type: ignore
            data_df[feature_combination + '_wordocc_right'] = data_df[feature_combination + '_wordocc_right'].str.strip()# type: ignore

        return data_df


    def transform_columns_to_wordcount(data_df, feature_combinations, test_df):
        words = {}

        for feature_combination in feature_combinations:

            # build relevant strings for vocabulary
            all_left_strings = data_df[['id_left', feature_combination + '_wordocc_left']].copy()
            all_left_strings = all_left_strings.rename(
                columns={'id_left': 'id', feature_combination + '_wordocc_left': feature_combination})
            all_right_strings = data_df[['id_right', feature_combination + '_wordocc_right']].copy()
            all_right_strings = all_right_strings.rename(
                columns={'id_right': 'id', feature_combination + '_wordocc_right': feature_combination})
            all_unique_strings = pd.concat([all_left_strings, all_right_strings])
            all_unique_strings = all_unique_strings.drop_duplicates(subset='id')

            # learn vocabulary
            count_vectorizer = CountVectorizer(min_df=2, binary=True)
            count_vectorizer.fit(all_unique_strings[feature_combination])

            words[feature_combination] = count_vectorizer.get_feature_names_out().tolist()

            # apply binary word occurrence
            left_matrix = count_vectorizer.transform(data_df[feature_combination + '_wordocc_left'])
            right_matrix = count_vectorizer.transform(data_df[feature_combination + '_wordocc_right'])
            data_df[feature_combination + '_wordocc_left'] = [x for x in left_matrix] # type: ignore
            data_df[feature_combination + '_wordocc_right'] = [x for x in right_matrix] # type: ignore

            if not isinstance(test_df, type(None)):
                left_matrix_test = count_vectorizer.transform(test_df[feature_combination + '_wordocc_left'])
                right_matrix_test = count_vectorizer.transform(test_df[feature_combination + '_wordocc_right'])
                test_df[feature_combination + '_wordocc_left'] = [x for x in left_matrix_test]# type: ignore
                test_df[feature_combination + '_wordocc_right'] = [x for x in right_matrix_test]# type: ignore

        return data_df, test_df, words


    def transform_columns_to_wordcooc(data_df, feature_combinations, test_df):
        for feature_combination in feature_combinations:
            data_df[feature_combination + '_wordcooc'] = list(
                map(lambda x, y: x.multiply(y).astype(int), data_df[feature_combination + '_wordocc_left'].values,
                    data_df[feature_combination + '_wordocc_right'].values))

            if not isinstance(test_df, type(None)):
                test_df[feature_combination + '_wordcooc'] = list(
                    map(lambda x, y: x.multiply(y).astype(int), test_df[feature_combination + '_wordocc_left'].values,
                        test_df[feature_combination + '_wordocc_right'].values))

        return data_df, test_df


    def preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name, valid_set=None,
                            test_set=None):
        columns_preprocess_wordcooc = [col + '_left' for col in columns_to_preprocess]
        columns_preprocess_wordcooc.extend([col + '_right' for col in columns_to_preprocess])

        main_df = process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations)

        if not isinstance(test_set, type(None)):
            test_df = process_df_columns_to_wordocc(test_set, columns_preprocess_wordcooc, feature_combinations)
        else:
            test_df = None

        main_df, test_df, words = transform_columns_to_wordcount(main_df, feature_combinations, test_df)
        main_df, test_df = transform_columns_to_wordcooc(main_df, feature_combinations, test_df)

        main_name = os.path.basename(file)
        new_main_name = main_name.replace('.pkl.gz', '_wordcooc')
        new_main_name = new_main_name.replace('.json.gz', '_wordcooc')

        out_path = f'data/processed/wordcooc/{experiment_name}/'

        os.makedirs(out_path + 'feature-names/', exist_ok=True)

        with open(out_path + 'feature-names/' + new_main_name + '_words.json', 'w') as f:
            json.dump(words, f, ensure_ascii=False)

        if isinstance(valid_set, type(None)):
            main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')# type: ignore
        else:
            if 'products' in file:
                validation_ids_df = pd.read_pickle(valid_set)
            else:
                validation_ids_df = pd.read_csv(valid_set)
            validation_df = main_df[main_df['pair_id'].isin(validation_ids_df['pair_id'].values)]# type: ignore

            main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')# type: ignore
            valid_name = new_main_name.replace('train', 'valid')
            validation_df.to_pickle(out_path + valid_name + '.pkl.gz', compression='gzip')

        if not isinstance(test_df, type(None)):
            test_name = os.path.basename(test_set)# type: ignore
            test_name = test_name.replace('.pkl.gz', '')
            test_name = test_name.replace('.json.gz', '')
            new_test_name = new_main_name + '_' + test_name

            test_df.to_pickle(out_path + new_test_name + '.pkl.gz', compression='gzip')


    for file in glob.glob('data/processed/training-sets/*'):
        if 'multi' in file or 'products' not in file:
            continue
            
        valid = file.replace('training', 'validation')
        valid = valid.replace('train', 'valid')
        if 'products' not in file:
            columns_to_preprocess = ['name', 'desc', 'brand']
            feature_combinations = ['name', 'brand+name', 'brand+name+desc']
            valid = valid.replace('.pkl.gz', '.csv')
            valid = valid.replace('preprocessed_', '')
            valid = valid.replace('interim', 'raw')
        else:
            columns_to_preprocess = ['name', 'desc', 'brand', 'price']
            feature_combinations = ['brand+name+price+desc']

        test_cat = os.path.basename(file).split('_')[1]
        test ='data/processed/gold-standards_adjusted/preprocessed_{}_gs.pkl.gz'.format(test_cat)
        print("File looked up is:", test, " and ", valid)
        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve_adjusted', valid_set=valid, test_set=test)
        
        test = test.replace('000un','050un')
        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve_adjusted', valid_set=valid, test_set=test)

        test = test.replace('050un','100un')
        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve_adjusted', valid_set=valid, test_set=test)

if __name__ == '__main__':
    #eprocess_data()
    #preprocess_data_for_ditto_hiergat()
    process_to_magellan()
    process_to_wordcooc()