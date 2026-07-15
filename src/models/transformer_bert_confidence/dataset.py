import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd

from pathlib import Path
import glob
import gzip
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

def assign_clusterid(identifier, product_id_dict, product_id_amount):
    try:
        result = product_id_dict[identifier]
    except KeyError:
        result = product_id_amount
    return result

def safe_text(x):
    if pd.isna(x) or x is None:
        return ""
    if isinstance(x, (float, int)):
        return str(x)
    if isinstance(x, (list, tuple)):
        return " ".join(map(str, x))
    return str(x)

def serialize_sample_lspc_contrastive(sample, only_name, use_price=False):
    string = ''
    name = safe_text(sample.get("name", ""))
    brand = safe_text(sample.get("brand", ""))
    desc = safe_text(sample.get("desc", ""))
    price = safe_text(sample.get("price", ""))

    if only_name:
        return f"{string} [COL] name [VAL] {' '.join(name.split(' ')[:50])}".strip()

    string = f"{string}[COL] brand [VAL] {' '.join(brand.split(' ')[:5])}".strip()
    string = f"{string} [COL] name [VAL] {' '.join(name.split(' ')[:50])}".strip()
    string = f"{string} [COL] price [VAL] {price}".strip()
    string = f"{string} [COL] description [VAL] {' '.join(desc.split(' ')[:100])}".strip()
    return string

def serialize_sample_lspc_pairwise(sample, side, only_name, use_price=False):
    
    string = ''
    name = safe_text(sample.get(f"name_{side}", ""))
    brand = safe_text(sample.get(f"brand_{side}", ""))
    desc = safe_text(sample.get(f"desc_{side}", ""))
    price = safe_text(sample.get(f"price_{side}", ""))

    if only_name:
        return f"{string} [COL] name [VAL] {' '.join(name.split(' ')[:50])}".strip()

    string = f"{string}[COL] brand [VAL] {' '.join(brand.split(' ')[:5])}".strip()
    string = f"{string} [COL] name [VAL] {' '.join(name.split(' ')[:50])}".strip()
    string = f"{string} [COL] price [VAL] {price}".strip()
    string = f"{string} [COL] description [VAL] {' '.join(desc.split(' ')[:100])}".strip()
    return string

class Augmenter():
    def __init__(self, aug):

        stopwords = ['[COL]', '[VAL]', 'name', 'desc', 'manufacturer', 'brand']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=0.1)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=0.1)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        aug = aug.strip('-')

        if aug == 'all':
            self.augs = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]
        
        if aug == 'typo':
            self.augs = [aug_typo, None]

        if aug == 'swap':
            self.augs = [aug_swap, None]

        if aug == 'delete':
            self.augs = [aug_del, None]

        if aug == 'crop':
            self.augs = [aug_crop, None]

        if aug == 'substitute':
            self.augs = [aug_sub, None]

        if aug == 'split':
            self.augs = [aug_split, None]

    def apply_aug(self, string):
        aug = random.choice(self.augs)
        if aug is None:
            return string
        else:
            return aug.augment(string)

class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='lspc', only_interm=False, aug=False, only_name=False):

        self.max_length = max_length
        if 'sigmod' not in dataset:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))

        self.dataset = dataset
        self.aug = aug
        self.only_name = only_name

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        data = pd.read_pickle(path, compression='gzip')
                
        if intermediate_set is not None:
            interm_data = pd.read_pickle(intermediate_set, compression='gzip')
            max_cid = data['product_id'].max()
            interm_data['product_id'] = interm_data['product_id']+ max_cid + 1
            if only_interm:
                data = interm_data
            else:
                data = data.append(interm_data)
        
        data = data.reset_index(drop=True)

        data = data.fillna('')

        data = self._prepare_data(data, True)
        
        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        selection = self.data[self.data['labels'] == example['labels']]
        # if len(selection) > 1:
        #     selection = selection.drop(idx)
        pos = selection.sample(1).iloc[0].copy()

        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])
            pos['features'] = self.augmenter.apply_aug(pos['features'])

        return (example, pos)

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        data['features'] = data.apply(serialize_sample_lspc_contrastive, args=(self.only_name, use_price), axis=1)

        label_enc = LabelEncoder()
        data['labels'] = label_enc.fit_transform(data['product_id'])

        self.label_encoder = label_enc

        data = data[['features', 'labels']]

        return data

class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_name=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_name = only_name

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path, compression='gzip')
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            raise ValueError("Only the released product benchmark is supported")

        if self.dataset_type != 'test':
            validation_set = pd.read_pickle((f'data/processed/validation-sets/preprocessed_{category}_valid_{size}.pkl.gz'), compression='gzip')
            if self.dataset_type != 'train':
                data = validation_set

        if additional_data is not None and self.dataset_type != 'test':
            add_data = pd.read_pickle(additional_data, compression='gzip')
            val_ids_add = pd.read_csv(f'{additional_data.replace("training-sets", "validation-sets").replace("_train.pkl.gz", "_valid.csv")}')


            if only_additional:
                if self.dataset_type == 'train':
                    data = add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])]
                else:
                    data = add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])]
            else:
                if self.dataset_type == 'train':
                    data = data.append(add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])])
                else:
                    data = data.append(add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])])

        data = data.fillna('')
        data = data.reset_index(drop=True)

        if 'products' in path:
            data = self._prepare_data(data, True)
        else:
            data = self._prepare_data(data, False)
        
        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(serialize_sample_lspc_pairwise, args=('left',self.only_name,use_price), axis=1)
            data['features_right'] = data.apply(serialize_sample_lspc_pairwise, args=('right',self.only_name,use_price), axis=1)

        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data

    

class BaselineClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_name=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_name = only_name

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path, compression='gzip')
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            data = pd.read_json(path, compression='gzip', lines=True)

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                validation_set = pd.read_pickle((f'data/processed/validation-sets/preprocessed_{category}_valid_{size}.pkl.gz'))

            if self.dataset_type != 'train':
                data = validation_set

        """if additional_data is not None and self.dataset_type != 'test':
            add_data = pd.read_pickle(additional_data)
            val_ids_add = pd.read_csv(f'{additional_data.replace("training-sets", "validation-sets").replace("_train.pkl.gz", "_valid.csv")}')

            if only_additional:
                if self.dataset_type == 'train':
                    data = add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])]
                else:
                    data = add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])]
            else:
                if self.dataset_type == 'train':
                    data = data.append(add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])])
                else:
                    data = data.append(add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])])"""

        data = data.fillna('')
        data = data.reset_index(drop=True)
        
        
        data = self._prepare_data(data, False)
        data["features_left"] = data["features_left"].apply(lambda x: str(x) if not isinstance(x, str) else x)
        data["features_right"] = data["features_right"].apply(lambda x: str(x) if not isinstance(x, str) else x)

        self.data = data
        

    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        
        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        left, right = example["features_left"], example["features_right"]

        # Normalize both sides to flat strings
        def flatten_text(x):
            if isinstance(x, str):
                return x
            if isinstance(x, (list, tuple)):
                return " ".join(map(str, x))
            return str(x)

        example['features_left'] = flatten_text(left)
        example['features_right'] = flatten_text(right)
        
        example_tokenized = self.tokenizer(example['features_left'], example['features_right'], padding=False, truncation='longest_first', max_length=self.max_length)
        example_tokenized['label'] = example['label']

        return example_tokenized

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(serialize_sample_lspc_pairwise, args=('left',self.only_name, use_price), axis=1)
            data['features_right'] = data.apply(serialize_sample_lspc_pairwise, args=('right',self.only_name, use_price), axis=1)

        data = data[['features_left', 'features_right', 'label']]

        return data

class BaselineMultiClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_name=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_name = only_name

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path, compression='gzip')
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            data = pd.read_json(path, lines=True)

        data = data.fillna('')
        data = data.reset_index(drop=True)

        data = self._prepare_data(data, True)
        
        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])

        example_tokenized = self.tokenizer(example['features'], padding=False, truncation='longest_first', max_length=self.max_length)
        example_tokenized['label'] = example['label']

        return example_tokenized

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc_contrastive, args=(self.only_name,use_price), axis=1)

        label_enc = LabelEncoder()
        data['label'] = label_enc.fit_transform(data['label'])

        self.label_encoder = label_enc

        data = data[['features', 'label']]

        return data
