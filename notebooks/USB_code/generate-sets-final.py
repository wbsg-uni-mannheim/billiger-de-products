#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from itertools import combinations
import math

from gensim.parsing.preprocessing import lower_to_unicode, preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric

import py_stringmatching as sm
from gensim.models import FastText
from copy import deepcopy

import pathlib
import torch
from tqdm.auto import tqdm
tqdm.pandas()

FASTTEXT_MODEL = FastText.load('../fasttext/deepmatcher_product_datasets.model').wv


# # Functions
# function that selects hard negatives in DBSCAN clusters using SoftTFIDF and Cosine similarity
from bdb import set_trace


def select_cc_clusters(corpus_dbscan, cc_candidates_seen):
    
    corpus_dbscan = corpus_dbscan.set_index('product_id', drop=False).copy()
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]
    
    selection_final = []
    selection_final_tupled = []
    #amounts = [400, 250, 100]
    large = 400
    medium = 250
    small = 100
    
    random.shuffle(cc_candidates_seen)
    

    sub_corpus_dbscan = corpus_dbscan.copy()
    cur_selection = set()
    cur_selection_tupled = set()
    i = 0

    while len(cur_selection) < large:
        for dbscan_id in cc_candidates_seen:
            sub_corpus = sub_corpus_dbscan[sub_corpus_dbscan['dbscan_cluster'] == dbscan_id].drop_duplicates('product_id').copy()
            if len(sub_corpus) < 5: #change to 2 or 3 -test and try
                continue
            
            print("Item is being looked at: ", i)
            sub_corpus['name_processed'] = sub_corpus['name'].apply(lower_to_unicode)
            sub_corpus['name_processed'] = sub_corpus['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))

            similarities=[#sm.SoftTfIdf(corpus_list=sub_corpus['name_processed'].tolist(), threshold=0.7), 
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]

            sub_cids = sorted(list(set(sub_corpus['product_id'])))
            random_pick = random.sample(sub_cids, 1)
            similarities_results = []
            sub_selection = set()

            example = sub_corpus.loc[random_pick]['name_processed'].iloc[0]
            sub_corpus = sub_corpus.drop(random_pick)

            for sim in similarities:
                try:
                    result = sub_corpus['name_processed'].apply(lambda x: sim.get_sim_score(example, x))
                except AttributeError:
                    try:
                        result = sub_corpus['name_processed'].apply(lambda x: sim.get_raw_score(example, x))
                    except AttributeError:
                        result = sub_corpus['name_processed'].apply(lambda x: sim.n_similarity(example, x))
                
                similarities_results.append(result)
            sorted_sim = [x.sort_values(ascending=False) for x in similarities_results]

            while len(sub_selection) <= 3:
                index = random.sample(range(4), 1)[0]
                found = False
                counter = 0
                while not found:
                    selected = sorted_sim[index].index[counter]
                    if selected not in sub_selection:
                        found = True
                    counter += 1
                sub_selection.add(selected)

            sub_selection.update(random_pick)
            cur_selection.update(sub_selection)
            cur_selection_tupled.add(tuple(sub_selection))

            sub_corpus_dbscan = sub_corpus_dbscan.loc[~sub_corpus_dbscan.index.isin(cur_selection)]
            i = i+1
            if len(cur_selection) == small:
                cur_selection_small = deepcopy(cur_selection)
                cur_selection_tupled_small = deepcopy(cur_selection_tupled)
            if len(cur_selection) == medium:
                cur_selection_medium = deepcopy(cur_selection)
                cur_selection_tupled_medium = deepcopy(cur_selection_tupled)
            if len(cur_selection) == large:
                cur_selection_large = cur_selection
                cur_selection_tupled_large = deepcopy(cur_selection_tupled)
                break

    selection_final.append(cur_selection_large)
    selection_final.append(cur_selection_medium)
    selection_final.append(cur_selection_small)
    selection_final_tupled.append(cur_selection_tupled_large)
    selection_final_tupled.append(cur_selection_tupled_medium)
    selection_final_tupled.append(cur_selection_tupled_small)
        
    return selection_final_tupled

# function that selects random clusters
def select_rnd_clusters(corpus_dbscan, ccs, cc_candidates_seen):

    ccs_set = set()
    ccs_set.update(*ccs)

    corpus_dbscan = corpus_dbscan[~corpus_dbscan['product_id'].isin(ccs_set)].copy()
    corpus_dbscan = corpus_dbscan[corpus_dbscan['dbscan_cluster'].isin(cc_candidates_seen)]
    
    selection_final = []
    #amounts = [400, 250, 100]
    large = 400
    medium = 250
    small = 100
    
    sub_corpus = corpus_dbscan.set_index('product_id', drop=False).copy()

    #counts = sub_corpus['product_id'].value_counts()
    #counts = counts[counts >6]
    #sub_corpus = sub_corpus[sub_corpus['product_id'].isin(counts.index)]

    sub_corpus = sub_corpus.sort_values('id')
    sub_corpus = sub_corpus.set_index('id', drop=False)
    sub_corpus = sub_corpus.drop_duplicates('product_id')
    rnd_cids = sorted(list(set(sub_corpus['product_id'])))

    sample = random.sample(rnd_cids, large)
    
    sub_sample_large = sample[:large]
    sub_sample_medium = sub_sample_large[:medium]
    sub_sample_small = sub_sample_medium[:small]
    sub_sample_large = set(sub_sample_large)
    sub_sample_medium = set(sub_sample_medium)
    sub_sample_small = set(sub_sample_small)
    
    selection_final.append(sub_sample_large)
    selection_final.append(sub_sample_medium)
    selection_final.append(sub_sample_small)
    
    return selection_final

# funtion to build pairs from the created train, validation and test splits
def build_pairs(corpus, ccs, rnd, ccs_unseen, rnd_unseen, ccs_val, rnd_val):
    
    counting = 0
    
    # seen
    ccs_set = set()
    ccs_set.update(*ccs)
    
    combined_cids = list(ccs_set | rnd)
    corpus_dbscan = corpus.sort_values('id').copy()
    corpus_dbscan = corpus_dbscan.set_index('product_id', drop=False)
    
    corpus_dbscan = corpus_dbscan.loc[combined_cids]
    corpus_dbscan = corpus_dbscan.set_index('id', drop=False)
    
    sample_to_10 = set()
    test_ids = set()
    valid_ids = set()
    train_ids = set()
    
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]
    
    
    
    corpus_dbscan['name_processed'] = corpus_dbscan['name'].apply(lower_to_unicode)
    corpus_dbscan['name_processed'] = corpus_dbscan['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    similarities=[#sm.SoftTfIdf(corpus_list=corpus_dbscan['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    
    for name, group in corpus_dbscan.groupby('product_id'):
        
        
        
        if len(group) > 7:
            max_len = len(group)
            if max_len < 15:
                cur_10 = sorted(list(set(group.sample(max_len)['id'])))
            else:
                if len(group) > 15:
                    counting +=1
                cur_10 = sorted(list(set(group.sample(15)['id'])))
        else:
            cur_10 = sorted(list(set(group.sample(7)['id'])))
        sample_to_10.update(cur_10)
        
        group = group[group['id'].isin(sample_to_10)].copy()
        group_ids = group['id']
        combs = list(combinations(group_ids, 2))
        sims = []
        for simno, sim in enumerate(similarities):
            cur_res = []
            for combination in combs:
                try:
                    result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                except AttributeError:
                    try:
                        result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                cur_res.append(result)
            sims.append(cur_res)
        #averaged = [sum(i)/len(similarities) for i in zip(*sims)]
        
        tupled = set()
        for sim in sims:
            tupled.update([(c, sim[i]) for i, c in enumerate(combs)])
                  
        sorted_sim = sorted(list(tupled), key = lambda y: y[1])
        sorted_sim = [x[0] for x in sorted_sim]
        check_cc_set = set()
        for cur_set in ccs:
            check_cc_set.update(cur_set)
        is_cc = name in check_cc_set
        
        if is_cc:
            cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
            rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]
            
            test_ids.update(set(*random.sample(cc_bucket, 1)))
            cc_bucket = [x for x in cc_bucket if x[0] not in test_ids and x[1] not in test_ids] 
            try:
                valid_ids.update(set(*random.sample(cc_bucket, 1)))
            except ValueError:
                valid_ids.update(set(*random.sample(rest_bucket, 1)))
            remaining = set(group_ids) - test_ids - valid_ids
            train_ids.update(remaining)

        else:
            test_ids.update(set(random.sample(cur_10, 2)))
            remaining = sorted(list(set(cur_10) - test_ids))
            valid_ids.update(set(random.sample(remaining, 2)))
            remaining = set(remaining) - valid_ids
            train_ids.update(remaining)

    assert len(test_ids & valid_ids & train_ids) == 0

    corpus_dbscan = corpus_dbscan[corpus_dbscan['id'].isin(sample_to_10)]

    # unseen 50
    ccs_seen_50 = random.sample(list(ccs), math.ceil(len(ccs)/2))
    rnd_seen_50 = random.sample(list(rnd), math.ceil(len(rnd)/2))
    ccs_unseen_50 = random.sample(list(ccs_unseen), math.ceil(len(ccs_unseen)/2))
    rnd_unseen_50 = random.sample(list(rnd_unseen), math.ceil(len(rnd_unseen)/2))
    
    ccs_val_50 = random.sample(list(ccs_val), math.ceil(len(ccs_val)/2))
    rnd_val_50 = random.sample(list(rnd_val), math.ceil(len(rnd_val)/2))
    
    ccs_seen_50_set = set()
    ccs_seen_50_set.update(*ccs_seen_50)
    rnd_seen_50_set = set()
    rnd_seen_50_set.update(rnd_seen_50)
    ccs_unseen_50_set = set()
    ccs_unseen_50_set.update(*ccs_unseen_50)
    rnd_unseen_50_set = set()
    rnd_unseen_50_set.update(rnd_unseen_50)
    
    ccs_val_50_set = set()
    ccs_val_50_set.update(*ccs_val_50)
    rnd_val_50_set = set()
    rnd_val_50_set.update(rnd_val_50)
    
    combined_cids_unseen_50 = ccs_seen_50_set | rnd_seen_50_set | ccs_unseen_50_set | rnd_unseen_50_set
    corpus_dbscan_unseen_50 = corpus.sort_values('id').copy()
    corpus_dbscan_unseen_50 = corpus_dbscan_unseen_50.set_index('product_id', drop=False)
    
    corpus_dbscan_unseen_50 = corpus_dbscan_unseen_50.loc[list(combined_cids_unseen_50)]
    corpus_dbscan_unseen_50 = corpus_dbscan_unseen_50.set_index('id', drop=False)
    
    sample_to_3_unseen_50 = set()
    test_ids_unseen_50 = set()
    
    corpus_dbscan_unseen_50['name_processed'] = corpus_dbscan_unseen_50['name'].apply(lower_to_unicode)
    corpus_dbscan_unseen_50['name_processed'] = corpus_dbscan_unseen_50['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    similarities=[#sm.SoftTfIdf(corpus_list=corpus_dbscan_unseen_50['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    
    for name, group in corpus_dbscan_unseen_50.groupby('product_id'):
        
        group_ids = group['id']
        combs = list(combinations(group_ids, 2))
        sims = []
        for simno, sim in enumerate(similarities):
            cur_res = []
            for combination in combs:
                try:
                    result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                except AttributeError:
                    try:
                        result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                cur_res.append(result)
            sims.append(cur_res)
        
        tupled = set()
        for sim in sims:
            tupled.update([(c, sim[i]) for i, c in enumerate(combs)])
                  
        sorted_sim = sorted(list(tupled), key = lambda y: y[1])
        sorted_sim = [x[0] for x in sorted_sim]
        check_cc_set = set()
        for cur_set in ccs_unseen:
            check_cc_set.update(cur_set)
        is_cc = name in check_cc_set
        
        if is_cc:
            
            cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
            rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]
            
            cur_3_unseen_50 = set(*random.sample(cc_bucket, 1))
            sample_to_3_unseen_50.update(cur_3_unseen_50)
        else:
            cur_3_unseen_50 = sorted(list(set(group.sample(2)['id'])))
            sample_to_3_unseen_50.update(cur_3_unseen_50)
        
#         cur_3_unseen_50 = sorted(list(set(group.sample(2)['id'])))
#         sample_to_3_unseen_50.update(cur_3_unseen_50)
        
        test_ids_unseen_50.update(sample_to_3_unseen_50)

    corpus_dbscan_unseen_50 = corpus_dbscan_unseen_50[corpus_dbscan_unseen_50['id'].isin(sample_to_3_unseen_50)]
    
    combined_cids_val_50 = ccs_seen_50_set | rnd_seen_50_set | ccs_val_50_set | rnd_val_50_set
    corpus_dbscan_val_50 = corpus.sort_values('id').copy()
    corpus_dbscan_val_50 = corpus_dbscan_val_50.set_index('product_id', drop=False)
    
    corpus_dbscan_val_50 = corpus_dbscan_val_50.loc[list(combined_cids_val_50)]
    corpus_dbscan_val_50 = corpus_dbscan_val_50.set_index('id', drop=False)
    
    sample_to_3_val_50 = set()
    test_ids_val_50 = set()
    
    corpus_dbscan_val_50['name_processed'] = corpus_dbscan_val_50['name'].apply(lower_to_unicode)
    corpus_dbscan_val_50['name_processed'] = corpus_dbscan_val_50['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    similarities=[#sm.SoftTfIdf(corpus_list=corpus_dbscan_val_50['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    
    for name, group in corpus_dbscan_val_50.groupby('product_id'):
        
        group_ids = group['id']
        combs = list(combinations(group_ids, 2))
        sims = []
        for simno, sim in enumerate(similarities):
            cur_res = []
            for combination in combs:
                try:
                    result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                except AttributeError:
                    try:
                        result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                cur_res.append(result)
            sims.append(cur_res)
            
        tupled = set()
        for sim in sims:
            tupled.update([(c, sim[i]) for i, c in enumerate(combs)])
                  
        sorted_sim = sorted(list(tupled), key = lambda y: y[1])
        sorted_sim = [x[0] for x in sorted_sim]
        check_cc_set = set()
        for cur_set in ccs_val:
            check_cc_set.update(cur_set)
        is_cc = name in check_cc_set
        if is_cc:
            
            cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
            rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]
            
            cur_3_val_50 = set(*random.sample(cc_bucket, 1))
            sample_to_3_val_50.update(cur_3_val_50)
            
        else:
            cur_3_val_50 = sorted(list(set(group.sample(2)['id'])))
            sample_to_3_val_50.update(cur_3_val_50)
        
#         cur_3_val_50 = sorted(list(set(group.sample(2)['id'])))
#         sample_to_3_val_50.update(cur_3_val_50)
        
        test_ids_val_50.update(sample_to_3_val_50)

    corpus_dbscan_val_50 = corpus_dbscan_val_50[corpus_dbscan_val_50['id'].isin(sample_to_3_val_50)]
    
     # unseen 100
    ccs_unseen_set = set()
    ccs_unseen_set.update(*ccs_unseen)
    
    combined_cids_unseen_100 = list(ccs_unseen_set | rnd_unseen)
    corpus_dbscan_unseen_100 = corpus.sort_values('id').copy()
    corpus_dbscan_unseen_100 = corpus_dbscan_unseen_100.set_index('product_id', drop=False)
    
    corpus_dbscan_unseen_100 = corpus_dbscan_unseen_100.loc[list(combined_cids_unseen_100)]
    corpus_dbscan_unseen_100 = corpus_dbscan_unseen_100.set_index('id', drop=False)
    
    sample_to_3_unseen_100 = set()
    test_ids_unseen_100 = set()
    
    corpus_dbscan_unseen_100['name_processed'] = corpus_dbscan_unseen_100['name'].apply(lower_to_unicode)
    corpus_dbscan_unseen_100['name_processed'] = corpus_dbscan_unseen_100['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    similarities=[#sm.SoftTfIdf(corpus_list=corpus_dbscan_unseen_100['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    
    budget_cc = 0
    budgetrnd = 0
    
    for name, group in corpus_dbscan_unseen_100.groupby('product_id'):
        
        group_ids = group['id']
        combs = list(combinations(group_ids, 2))
        sims = []
        for simno, sim in enumerate(similarities):
            cur_res = []
            for combination in combs:
                try:
                    result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                except AttributeError:
                    try:
                        result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                cur_res.append(result)
            sims.append(cur_res)
            
        tupled = set()
        for sim in sims:
            tupled.update([(c, sim[i]) for i, c in enumerate(combs)])
                  
        sorted_sim = sorted(list(tupled), key = lambda y: y[1])
        sorted_sim = [x[0] for x in sorted_sim]
        check_cc_set = set()
        for cur_set in ccs_unseen:
            check_cc_set.update(cur_set)
        is_cc = name in check_cc_set
        if is_cc:
            
            cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
            rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]
            
            cur_3_unseen_100 = set(*random.sample(cc_bucket, 1))
            sample_to_3_unseen_100.update(cur_3_unseen_100)

        else:
            cur_3_unseen_100 = sorted(list(set(group.sample(2)['id'])))
            sample_to_3_unseen_100.update(cur_3_unseen_100)
        
#         cur_3_unseen_100 = sorted(list(set(group.sample(2)['id'])))
#         sample_to_3_unseen_100.update(cur_3_unseen_100)
        
        test_ids_unseen_100.update(sample_to_3_unseen_100)

    corpus_dbscan_unseen_100 = corpus_dbscan_unseen_100[corpus_dbscan_unseen_100['id'].isin(sample_to_3_unseen_100)]
    
    ccs_val_set = set()
    ccs_val_set.update(*ccs_unseen)
    
    combined_cids_val_100 = list(ccs_val_set | rnd_val)
    corpus_dbscan_val_100 = corpus.sort_values('id').copy()
    corpus_dbscan_val_100 = corpus_dbscan_val_100.set_index('product_id', drop=False)
    
    corpus_dbscan_val_100 = corpus_dbscan_val_100.loc[combined_cids_val_100]
    corpus_dbscan_val_100 = corpus_dbscan_val_100.set_index('id', drop=False)
    
    sample_to_3_val_100 = set()
    test_ids_val_100 = set()
    
    corpus_dbscan_val_100['name_processed'] = corpus_dbscan_val_100['name'].apply(lower_to_unicode)
    corpus_dbscan_val_100['name_processed'] = corpus_dbscan_val_100['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    similarities=[#sm.SoftTfIdf(corpus_list=corpus_dbscan_val_100['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    
    budget_cc = 0
    budgetrnd = 0
    
    for name, group in corpus_dbscan_val_100.groupby('product_id'):
        
        group_ids = group['id']
        combs = list(combinations(group_ids, 2))
        sims = []
        for simno, sim in enumerate(similarities):
            cur_res = []
            for combination in combs:
                try:
                    result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                except AttributeError:
                    try:
                        result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                cur_res.append(result)
            sims.append(cur_res)
        tupled = set()
        for sim in sims:
            tupled.update([(c, sim[i]) for i, c in enumerate(combs)])
                  
        sorted_sim = sorted(list(tupled), key = lambda y: y[1])
        sorted_sim = [x[0] for x in sorted_sim]
        check_cc_set = set()
        for cur_set in ccs_val:
            check_cc_set.update(cur_set)
        is_cc = name in check_cc_set
        if is_cc:
            
            cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
            rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]
            
            cur_3_val_100 = set(*random.sample(cc_bucket, 1))
            sample_to_3_val_100.update(cur_3_val_100)
            
        else:
            cur_3_val_100 = sorted(list(set(group.sample(2)['id'])))
            sample_to_3_val_100.update(cur_3_val_100)
        
#         cur_3_val_100 = sorted(list(set(group.sample(2)['id'])))
#         sample_to_3_val_100.update(cur_3_val_100)
        
        test_ids_val_100.update(sample_to_3_val_100)

    corpus_dbscan_val_100 = corpus_dbscan_val_100[corpus_dbscan_val_100['id'].isin(sample_to_3_val_100)]
    
    
    # build test, valid, train
    test_set, test_ccs = build_test(corpus_dbscan[corpus_dbscan['id'].isin(test_ids)], test_ids)
    try:
        assert len(test_set) == 9* len(combined_cids)
    except AssertionError:
        set_trace()

    test_set_unseen_50, test_ccs_50 = build_test(corpus_dbscan_unseen_50[corpus_dbscan_unseen_50['id'].isin(test_ids_unseen_50)], test_ids_unseen_50)
    try:
        assert len(test_set_unseen_50) == 9* len(combined_cids)
    except AssertionError:
        set_trace()
        
    test_set_unseen_100, test_ccs_100 = build_test(corpus_dbscan_unseen_100[corpus_dbscan_unseen_100['id'].isin(test_ids_unseen_100)], test_ids_unseen_100)
    try:
        assert len(test_set_unseen_100) == 9* len(combined_cids)
    except AssertionError:
        set_trace()
        
    print('Test set built')
    
    valid_small, valid_medium, valid_large, valid_ccs = build_train(corpus_dbscan[corpus_dbscan['id'].isin(valid_ids)], valid_ids, ccs, is_valid=True)
    try:
        assert len(valid_large) == 9* len(combined_cids) and len(valid_medium) == 7* len(combined_cids) and len(valid_small) == 5* len(combined_cids)
    except AssertionError:
        set_trace()
        
    valid_small_unseen_50, valid_medium_unseen_50, valid_large_unseen_50, valid_ccs_unseen_50 = build_train(corpus_dbscan_val_50[corpus_dbscan_val_50['id'].isin(test_ids_val_50)], test_ids_val_50, ccs | ccs_val, is_valid=True)
    try:
        assert len(valid_large_unseen_50) == 9* len(combined_cids) and len(valid_medium_unseen_50) == 7* len(combined_cids) and len(valid_small_unseen_50) == 5* len(combined_cids)
    except AssertionError:
        set_trace()
        
    valid_small_unseen_100, valid_medium_unseen_100, valid_large_unseen_100, valid_ccs_unseen_100 = build_train(corpus_dbscan_val_100[corpus_dbscan_val_100['id'].isin(test_ids_val_100)], test_ids_val_100, ccs | ccs_val, is_valid=True)
    try:
        assert len(valid_large_unseen_100) == 9* len(combined_cids) and len(valid_medium_unseen_100) == 7* len(combined_cids) and len(valid_small_unseen_100) == 5* len(combined_cids)
    except AssertionError:
        set_trace()
        
        
    print('Validation set built')
        
    train_small, train_medium, train_large, train_ccs = build_train(corpus_dbscan[corpus_dbscan['id'].isin(train_ids)], train_ids, ccs)
    try:
        assert len(train_medium) == 12* len(combined_cids) and len(train_small) == 5* len(combined_cids)
    except AssertionError:
        set_trace()

    print('Train set built')
    
    ccs = [train_ccs, (valid_ccs, valid_ccs_unseen_50, valid_ccs_unseen_100), (test_ccs, test_ccs_50, test_ccs_100)]
    
    print(f'Counted {counting} Cluster larger than 15')
        
    return (train_small, train_medium, train_large), ((valid_small, valid_medium, valid_large),(valid_small_unseen_50, valid_medium_unseen_50, valid_large_unseen_50),(valid_small_unseen_100, valid_medium_unseen_100, valid_large_unseen_100)), (test_set, test_set_unseen_50, test_set_unseen_100), ccs

# function to build the three development set sizes from the train or validation split
def build_train(corpus, ids, ccs, is_valid=False):
    
    corpus = corpus.copy()
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]
    cids = set(corpus['product_id'])
    
    corpus['name_processed'] = corpus['name'].apply(lower_to_unicode).copy()
    corpus['name_processed'] = corpus['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    all_pos = set()
    all_neg = set()
    
    all_pos_small = set()
    all_neg_small = set()
    
    all_pos_medium = set()
    all_neg_medium = set()
    
    small = set()
    medium = set()
    large = set()
    
    small_ccs = set()
    medium_ccs = set()
    large_ccs = set()
    
    similarities=[sm.SoftTfIdf(corpus_list=corpus['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
    budget_cc = 0
    budgetrnd = 0
    
    for cid in cids:
        
        cur_small = set()
        cur_medium = set()
        cur_large = set()
        
        sub_corpus = corpus[corpus['product_id'] == cid]
        sub_corpus_wo = corpus[~(corpus['product_id'] == cid)]
        
        if not is_valid:
            group = sub_corpus.copy()
            group_ids = group['id']
            combs = list(combinations(group_ids, 2))
            sims = []
            for simno, sim in enumerate(similarities):
                cur_res = []
                for combination in combs:
                    try:
                        result = sim.get_sim_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    except AttributeError:
                        try:
                            result = sim.get_raw_score(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                        except AttributeError:
                            result = sim.n_similarity(group[group['id'] == combination[0]].iloc[0]['name_processed'], group[group['id'] == combination[1]].iloc[0]['name_processed'])
                    cur_res.append(result)
                sims.append(cur_res)
            tupled = set()
            for sim in sims:
                tupled.update([(c, sim[i]) for i, c in enumerate(combs)])

            sorted_sim = sorted(list(tupled), key = lambda y: y[1])
            sorted_sim = [x[0] for x in sorted_sim]
            check_cc_set = set()
            for cur_set in ccs:
                check_cc_set.update(cur_set)
            is_cc = cid in check_cc_set

            if is_cc:

                cc_bucket = sorted_sim[:(math.ceil(len(sorted_sim)/5))]
                rest_bucket = sorted_sim[(math.ceil(len(sorted_sim)/5)):]

                first = set(*random.sample(cc_bucket, 1))
                cur_small.update(first)
                cur_medium.update(first)
                cur_large.update(first)
                small.update(first)
                medium.update(first)
                large.update(first)

            else:
                first = set(random.sample(group_ids.tolist(), 2))
                cur_small.update(first)
                cur_medium.update(first)
                cur_large.update(first)
                small.update(first)
                medium.update(first)
                large.update(first)
        
        for idx, (i, row) in enumerate(sub_corpus.sample(frac=1.0).iterrows()):
            if is_valid:
                if idx < 2:
                    small.add(i)
                if idx < 2:
                    medium.add(i)
                if idx < 2:
                    large.add(i)
            else:
#                 if idx < 2:
#                     small.add(i)
                if len(cur_medium) < 3:
                    cur_medium.add(i)
                    medium.add(i)
                if len(cur_large) < 11:
                    cur_large.add(i)
                    large.add(i)
    
        if not is_valid:
            assert len(cur_small) == 2
            assert len(cur_medium) == 3
            assert len(cur_large) >= 3
        
    for cid in tqdm(cids):
        
        sub_corpus = corpus[corpus['product_id'] == cid]
        sub_corpus_wo = corpus[~(corpus['product_id'] == cid)]
        
        
        for num, size in enumerate([small, medium, large]):
            
            corpus_current = sub_corpus[sub_corpus['id'].isin(size)]
            corpus_current_wo = sub_corpus_wo[sub_corpus_wo['id'].isin(size)]
            
            similarities=[sm.SoftTfIdf(corpus_list=corpus['name_processed'].tolist(), threshold=0.7),
                          sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
                              
            
            for i, row in corpus_current.iterrows():
                
                example = row['name_processed']
                similarities_results = []
                
                
                for sim in similarities:
                    try:
                        result = corpus_current_wo['name_processed'].apply(lambda x: sim.get_sim_score(example, x))
                    except AttributeError:
                        try:
                            result = corpus_current_wo['name_processed'].apply(lambda x: sim.get_raw_score(example, x))
                        except AttributeError:
                            result = corpus_current_wo['name_processed'].apply(lambda x: sim.n_similarity(example, x))
                    similarities_results.append(result)
                sorted_sim = [x.sort_values(ascending=False) for x in similarities_results]

                selected_negs = set()
                already_selected_clusters = set()
                ids_to_remove = set()
                
                if num == 0:
                    limit = 1
                elif num == 1:
                    limit = 2
                else:
                    limit = 3
                    
                while len(selected_negs) < limit:

                    index = random.sample(range(4), 1)[0]
                    found = False
                    counter = 0
                    while not found:
                        selected = sorted_sim[index].index[counter]
                        rel_clu = sub_corpus_wo.loc[selected]['product_id']
                        if num == 0:
                            if rel_clu not in already_selected_clusters and (i,selected) not in all_neg_small and (selected,i) not in all_neg_small and (i,selected) not in selected_negs and (selected,i) not in selected_negs:
                                selected_negs.update(random.sample([(i,selected), (selected,i)],1))
                                ids_to_remove.add(selected)
                                already_selected_clusters.add(rel_clu)
                                found = True
                        elif num == 1:
                            if rel_clu not in already_selected_clusters and (i,selected) not in all_neg_medium and (selected,i) not in all_neg_medium and (i,selected) not in selected_negs and (selected,i) not in selected_negs:
                                selected_negs.update(random.sample([(i,selected), (selected,i)],1))
                                ids_to_remove.add(selected)
                                already_selected_clusters.add(rel_clu)
                                found = True
                        else:
                            if rel_clu not in already_selected_clusters and (i,selected) not in all_neg and (selected,i) not in all_neg and (i,selected) not in selected_negs and (selected,i) not in selected_negs:
                                selected_negs.update(random.sample([(i,selected), (selected,i)],1))
                                ids_to_remove.add(selected)
                                already_selected_clusters.add(rel_clu)
                                found = True
                        counter += 1
                            
                if num == 0:
                    all_neg_small.update(selected_negs)
                    small_ccs.update(selected_negs)
                    cur_length = len(all_neg_small)
                    cur_selection = all_neg_small
                elif num == 1:
                    all_neg_medium.update(selected_negs)
                    medium_ccs.update(selected_negs)
                    cur_length = len(all_neg_medium)
                    cur_selection = all_neg_medium
                elif num == 2:
                    all_neg.update(selected_negs)
                    large_ccs.update(selected_negs)
                    cur_length = len(all_neg)
                    cur_selection = all_neg

                rnd_sample_corpus = corpus_current_wo.loc[~corpus_current_wo.index.isin(ids_to_remove)]
                
                while len(cur_selection) == cur_length:
                    rnd_id = rnd_sample_corpus.sample(1)['id'].iloc[0]
                    rnd_pair = random.sample([(i, rnd_id), (rnd_id, i)], 1)
                    if (i, rnd_id) not in cur_selection and (rnd_id, i) not in cur_selection:
                        cur_selection.update(rnd_pair)
                        
                if num == 0:
                    all_neg_small.update(cur_selection)
                elif num == 1:
                    all_neg_medium.update(cur_selection)
                elif num == 2:
                    all_neg.update(cur_selection)
                    
        positives = list(combinations(sub_corpus['id'].tolist(), 2))
        positives_shuffled = [random.sample([(x[0],x[1]),(x[1], x[0])],1)[0] for x in positives]
        all_pos.update(positives_shuffled)
                
        positives_small = list(combinations(small, 2))
        positives_selected_small = [x for x in positives_shuffled if (x[0],x[1]) in positives_small or (x[1],x[0]) in positives_small]
        all_pos_small.update(positives_selected_small)
        
        positives_medium = list(combinations(medium, 2))
        positives_selected_medium = [x for x in positives_shuffled if (x[0],x[1]) in positives_medium or (x[1],x[0]) in positives_medium]
        all_pos_medium.update(positives_selected_medium)
        
    large = all_pos | all_neg
    medium = all_pos_medium | all_neg_medium
    small = all_pos_small | all_neg_small
    
    return small, medium, large, (small_ccs, medium_ccs, large_ccs)

# function to build the test split from the test offers
def build_test(corpus, ids):
    
    corpus = corpus.copy()
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]
    cids = set(corpus['product_id'])
    
    corpus['name_processed'] = corpus['name'].apply(lower_to_unicode).copy()
    corpus['name_processed'] = corpus['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
    
    all_pos = set()
    all_neg = set()
    
    test_ccs = set()
    
    for cid in tqdm(cids):
        sub_corpus = corpus[corpus['product_id'] == cid]
        sub_corpus_wo = corpus[~(corpus['product_id'] == cid)]
        
        similarities=[sm.SoftTfIdf(corpus_list=corpus['name_processed'].tolist(), threshold=0.7),
                     sm.Cosine(), 
                         sm.GeneralizedJaccard(threshold=0.7),
                         sm.Dice(),
                         FASTTEXT_MODEL]
        
        positives = list(combinations(sub_corpus['id'].tolist(), 2))
        positives_shuffled = [random.sample([(x[0],x[1]),(x[1], x[0])],1)[0] for x in positives]

        all_pos.update(positives_shuffled)
        
        for i, row in sub_corpus.iterrows():
            
            example = row['name_processed']
            similarities_results = []
            for sim in similarities:
                try:
                    result = sub_corpus_wo['name_processed'].apply(lambda x: sim.get_sim_score(example, x))
                except AttributeError:
                    try:
                        result = sub_corpus_wo['name_processed'].apply(lambda x: sim.get_raw_score(example, x))
                    except AttributeError:
                        result = sub_corpus_wo['name_processed'].apply(lambda x: sim.n_similarity(example, x))
                similarities_results.append(result)
            sorted_sim = [x.sort_values(ascending=False) for x in similarities_results]
            
            selected_negs = set()
            already_selected_clusters = set()
            ids_to_remove = set()
            
            while len(selected_negs) < 3:
                
                index = random.sample(range(4), 1)[0]
                found = False
                counter = 0
                while not found:
                    selected = sorted_sim[index].index[counter]
                    rel_clu = sub_corpus_wo.loc[selected]['product_id']
                    if rel_clu not in already_selected_clusters and (i,selected) not in all_neg and (selected,i) not in all_neg and (i,selected) not in selected_negs and (selected,i) not in selected_negs:
                        selected_negs.update(random.sample([(i,selected), (selected,i)],1))
                        ids_to_remove.add(selected)
                        already_selected_clusters.add(rel_clu)
                        found = True
                    counter += 1
                
            
            test_ccs.update(selected_negs)
            all_neg.update(selected_negs)
            cur_length = len(all_neg)
            rnd_sample_corpus = sub_corpus_wo.loc[~sub_corpus_wo.index.isin(ids_to_remove)]
            
            while len(all_neg) == cur_length:
                rnd_id = rnd_sample_corpus.sample(1)['id'].iloc[0]

                if (i, rnd_id) not in all_neg and (rnd_id, i) not in all_neg:
                    all_neg.update(random.sample([(i, rnd_id), (rnd_id, i)], 1))
            
    all_ids = all_pos | all_neg
    
    return all_ids, test_ccs

def check_if_hard(row, ccs):
    left = row['id_left']
    right = row['id_right']
    if row['label'] == 1:
        return False
    for cc_set in ccs:
        if left in cc_set and right in cc_set:
            return True
    return False

def generate_pairs(id_pairs, corpus, ccs):
    id_pairs = list(id_pairs)
    corpus = corpus[['id', 'brand', 'name', 'desc', 'price','product_id']].set_index('id', drop=False)
    left_ids, right_ids = list(zip(*id_pairs))

    left_offers = corpus.loc[left_ids,:]
    right_offers = corpus.loc[right_ids,:]
    
    left_offers = left_offers.reset_index(drop=True)
    right_offers = right_offers.reset_index(drop=True)
    
    joined = left_offers.join(right_offers, lsuffix='_left', rsuffix='_right')
    joined['pair_id'] = joined['id_left'].astype(str) + '#' + joined['id_right'].astype(str)
    joined['label'] = joined['product_id_left'] == joined['product_id_right']
    joined['label'] = joined['label'].astype(int)
    joined['is_hard_negative'] = joined.apply(check_if_hard, args=(ccs,), axis=1)
    return joined

def generate_multiclass(pairwise_set, corpus, unseen):
    ids = set()
    pairwise_set = pairwise_set[pairwise_set['label'] == 1]
    corpus =  corpus[['id', 'brand', 'name', 'desc', 'price','product_id']].set_index('id', drop=False)
    ids.update(pairwise_set['id_left'])
    ids.update(pairwise_set['id_right'])
    ids = list(ids)
    multiclass_set = corpus.loc[ids,:]
    multiclass_set['label'] = multiclass_set['product_id']
    multiclass_set['unseen'] = multiclass_set['product_id'].apply(lambda x: True if x in unseen else False)
    multiclass_set = multiclass_set.reset_index(drop=True)
    
    return multiclass_set


# # Load cleansed PDC2020 corpus and split into seen and unseen candidates given labeled DBSCAN clusters

corpus = pd.read_pickle("../data/working/dedup_preprocessed_rev2_docs_since_2020_01_01_only_de_strict_only_long_name.pkl.gz")
corpus = corpus[["id", "product_id", "name", "desc", "brand", "shop_cat", "price"]]
print(len(corpus))
corpus.head()
"""cats = [
    "Elektronik & Computer",
    "Auto & Motorrad",
    "Kleidung & Accessoires",
    "Spielzeug & Baby"
]

corpus = corpus[corpus["top_category_mapped"].isin(cats)].copy()"""


# Only include two sctriucture categories (with custom product IDS such as IDs for diferent types of fridges).
# - Elektronik & Computer (Most benchmarks are based on that)
# - Auto & Motorrad (contains lots of Reifen with clear numbers)
# 
# Two instructure categories such as:
# - Kleidung & Accessoires
# - Spielzeug & Baby

# # select hard negative clusters for all three difficulties, 80%, 50% and 20% as well as complentary random clusters

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces]


cc_candidates_seen = pd.read_csv('../data/working/dbscan/seen_dbscan_clusters.csv')
#cc_candidates_seen = cc_candidates_seen[cc_candidates_seen['good'] == 1]
cc_candidates_seen = sorted(list(set(cc_candidates_seen['dbscan_cluster'])))

cc_candidates_unseen = pd.read_csv('../data/working/dbscan/unseen_dbscan_clusters.csv')
#cc_candidates_unseen = cc_candidates_unseen[cc_candidates_unseen['good'] == 1]
cc_candidates_unseen = sorted(list(set(cc_candidates_unseen['dbscan_cluster'])))

dbscan_mapping = pd.read_csv('../data/working/dbscan/seen_dbscan_mapping.csv')
corpus_dbscan = corpus.merge(dbscan_mapping, how='outer', on='product_id').copy()

dbscan_mapping_unseen = pd.read_csv('../data/working/dbscan/unseen_dbscan_mapping.csv')
corpus_dbscan_unseen = corpus.merge(dbscan_mapping_unseen, how='outer', on='product_id').copy()

print("Seen corpus size before filtering:", corpus_dbscan.shape[0])
print("Unseen corpus size before filtering:", corpus_dbscan_unseen.shape[0])

corpus_dbscan = corpus_dbscan[corpus_dbscan['name'].apply(lambda x: isinstance(x, str))].copy() # new as we do not have all items, only a few categories
corpus_dbscan['name_processed'] = corpus_dbscan['name'].apply(lower_to_unicode)
corpus_dbscan['name_processed'] = corpus_dbscan['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
corpus_dbscan['name_processed'] = corpus_dbscan['name_processed'].apply(lambda x: ' '.join(x))
#corpus_dbscan = corpus_dbscan.drop_duplicates(subset='name_processed') # Not needed for clean data

corpus_dbscan_unseen = corpus_dbscan_unseen[corpus_dbscan_unseen['name'].apply(lambda x: isinstance(x, str))].copy()
corpus_dbscan_unseen.dropna(subset = ['name']) # new as we do not have all items, only a few categories
corpus_dbscan_unseen['name_processed'] = corpus_dbscan_unseen['name'].apply(lower_to_unicode)
corpus_dbscan_unseen['name_processed'] = corpus_dbscan_unseen['name_processed'].apply(preprocess_string, args=(CUSTOM_FILTERS,))
corpus_dbscan_unseen['name_processed'] = corpus_dbscan_unseen['name_processed'].apply(lambda x: ' '.join(x))
#corpus_dbscan_unseen = corpus_dbscan_unseen.drop_duplicates(subset='name_processed') # Not needed for clean data

counts = corpus_dbscan['product_id'].value_counts()
counts = counts[counts > 6] # kann man vielleicht auch absenken
corpus_dbscan = corpus_dbscan[corpus_dbscan['product_id'].isin(counts.index)]

counts_unseen = corpus_dbscan_unseen['product_id'].value_counts()
counts_unseen = counts_unseen[counts_unseen > 3]
counts_unseen = counts_unseen[counts_unseen < 7]
corpus_dbscan_unseen = corpus_dbscan_unseen[corpus_dbscan_unseen['product_id'].isin(counts_unseen.index)]

print("Seen corpus size after filtering:", corpus_dbscan.shape[0])
print("Unseen corpus size after filtering:", corpus_dbscan_unseen.shape[0])

print("candidates_seen sie:", len(cc_candidates_seen))
print("candidates amount of products:", len(set(corpus_dbscan[corpus_dbscan['dbscan_cluster'].isin(cc_candidates_seen)]['product_id'])))
print("cluster size seen:", corpus_dbscan[corpus_dbscan['dbscan_cluster'].isin(cc_candidates_seen)].groupby('dbscan_cluster').size().median())
unique_clusters_in_data = corpus_dbscan['dbscan_cluster'].nunique()
print("Unique clusters present in corpus:", unique_clusters_in_data)


ccs_large, ccs_medium, ccs_small = select_cc_clusters(corpus_dbscan, cc_candidates_seen)
print("Step 1 done")
ccs_large_unseen, ccs_medium_unseen, ccs_small_unseen = select_cc_clusters(corpus_dbscan_unseen, cc_candidates_unseen)
print("Step 2 done")
ccs_unpack_unseen_large = set()
ccs_unpack_unseen_large.update(*ccs_large_unseen)
print("Step 3 done")
rnd_large, rnd_medium, rnd_small = select_rnd_clusters(corpus_dbscan, ccs_large, cc_candidates_seen)
print("Step 4 done")
rnd_large_unseen, rnd_medium_unseen, rnd_small_unseen = select_rnd_clusters(corpus_dbscan_unseen, ccs_large_unseen, cc_candidates_unseen)
print("Step 5 done")
merged = ccs_unpack_unseen_large | rnd_large_unseen
print("Step 6 done")
corpus_dbscan_val = corpus_dbscan_unseen[~corpus_dbscan_unseen['product_id'].isin(merged)]
print("Step 7 done")
ccs_large_val, ccs_medium_val, ccs_small_val = select_cc_clusters(corpus_dbscan_val, cc_candidates_unseen)
print("Step 8 done")
rnd_large_val, rnd_medium_val, rnd_small_val = select_rnd_clusters(corpus_dbscan_val, ccs_large_val, cc_candidates_unseen)


#ensure larger sets contain all clusters from smaller sets
assert len(ccs_small & ccs_medium) == 20
assert len(ccs_small & ccs_large) == 20
assert len(ccs_medium & ccs_large) == 50

assert len(rnd_small & rnd_medium) == 100
assert len(rnd_small & rnd_large) == 100
assert len(rnd_medium & rnd_large) == 250

assert len(ccs_small_unseen & ccs_medium_unseen) == 20
assert len(ccs_small_unseen & ccs_large_unseen) == 20
assert len(ccs_medium_unseen & ccs_large_unseen) == 50

assert len(rnd_small_unseen & rnd_medium_unseen) == 100
assert len(rnd_small_unseen & rnd_large_unseen) == 100
assert len(rnd_medium_unseen & rnd_large_unseen) == 250

assert len(ccs_small_val & ccs_medium_val) == 20
assert len(ccs_small_val & ccs_large_val) == 20
assert len(ccs_medium_val & ccs_large_val) == 50

assert len(rnd_small_val & rnd_medium_val) == 100
assert len(rnd_small_val & rnd_large_val) == 100
assert len(rnd_medium_val & rnd_large_val) == 250

ccs_unpack_large = set()
ccs_unpack_large.update(*ccs_large)

ccs_unpack_medium = set()
ccs_unpack_medium.update(*ccs_medium)

ccs_unpack_small = set()
ccs_unpack_small.update(*ccs_small)

ccs_unpack_unseen_large = set()
ccs_unpack_unseen_large.update(*ccs_large_unseen)

ccs_unpack_unseen_medium = set()
ccs_unpack_unseen_medium.update(*ccs_medium_unseen)

ccs_unpack_unseen_small = set()
ccs_unpack_unseen_small.update(*ccs_small_unseen)

ccs_unpack_val_large = set()
ccs_unpack_val_large.update(*ccs_large_val)

ccs_unpack_val_medium = set()
ccs_unpack_val_medium.update(*ccs_medium_val)

ccs_unpack_val_small = set()
ccs_unpack_val_small.update(*ccs_small_val)

assert len((ccs_unpack_small | ccs_unpack_medium | ccs_unpack_large | rnd_small | rnd_medium | rnd_large) & (ccs_unpack_unseen_small | ccs_unpack_unseen_medium | ccs_unpack_unseen_large | rnd_small_unseen | rnd_medium_unseen | rnd_large_unseen)) == 0
assert len((ccs_unpack_small | ccs_unpack_medium | ccs_unpack_large | rnd_small | rnd_medium | rnd_large) & (ccs_unpack_val_small | ccs_unpack_val_medium | ccs_unpack_val_large | rnd_small_val | rnd_medium_val | rnd_large_val)) == 0
assert len((ccs_unpack_unseen_small | ccs_unpack_unseen_medium | ccs_unpack_unseen_large | rnd_small_unseen | rnd_medium_unseen | rnd_large_unseen) & (ccs_unpack_val_small | ccs_unpack_val_medium | ccs_unpack_val_large | rnd_small_val | rnd_medium_val | rnd_large_val)) == 0


# # Generate training, validation and test sets for all hardness levels, development sizes and unseen percentages
train_80, valid_80, test_80, ccs_80 = build_pairs(pd.concat([corpus_dbscan, corpus_dbscan_unseen]), ccs_large, rnd_small, ccs_large_unseen, rnd_small_unseen, ccs_large_val, rnd_small_val)
hard = ('80cc20rnd000un',[train_80, valid_80, test_80, ccs_80])

train_50, valid_50, test_50, ccs_50 = build_pairs(pd.concat([corpus_dbscan, corpus_dbscan_unseen]), ccs_medium, rnd_medium, ccs_medium_unseen, rnd_medium_unseen, ccs_medium_val, rnd_medium_val)
medium = ('50cc50rnd000un',[train_50, valid_50, test_50, ccs_50])

train_20, valid_20, test_20, ccs_20 = build_pairs(pd.concat([corpus_dbscan, corpus_dbscan_unseen]), ccs_small, rnd_large, ccs_small_unseen, rnd_large_unseen, ccs_small_val, rnd_large_val)
easy = ('20cc80rnd000un',[train_20, valid_20, test_20, ccs_20])


unseen = ccs_unpack_unseen_large | ccs_unpack_val_large | rnd_large_unseen |rnd_large_val


# # Materialize datasets and write them to file

paths = ['training-sets', 'validation-sets', 'gold-standards']

for cur_path in paths:
    path = pathlib.Path(f'../data/derived/{cur_path}/')
    path.mkdir(parents=True, exist_ok=True)

for combination in [hard, medium, easy]:
    name = combination[0]
    train, valid, test, ccs = combination[1]
    
    valid_050_unseen = valid[1]
    valid_100_unseen = valid[2]
    valid = valid[0]
    
    train_ccs = ccs[0]
    valid_ccs = ccs[1][0]
    valid_ccs_050_unseen = ccs[1][1]
    valid_ccs_100_unseen = ccs[1][2]
    test_ccs = ccs[2]
    
    test_000_unseen = generate_pairs(test[0], corpus, test_ccs[0]).sample(frac=1.0, random_state=42)
    test_050_unseen = generate_pairs(test[1], corpus, test_ccs[1]).sample(frac=1.0, random_state=42)
    test_100_unseen = generate_pairs(test[2], corpus, test_ccs[2]).sample(frac=1.0, random_state=42)

    test_multi_000_unseen = generate_multiclass(test_000_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    test_multi_050_unseen = generate_multiclass(test_050_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    test_multi_100_unseen = generate_multiclass(test_100_unseen, corpus, unseen).sample(frac=1.0, random_state=42)

    train_small = generate_pairs(train[0], corpus, train_ccs[0]).sample(frac=1.0, random_state=42)
    train_medium = generate_pairs(train[1], corpus, train_ccs[1]).sample(frac=1.0, random_state=42)
    train_large = generate_pairs(train[2], corpus, train_ccs[2]).sample(frac=1.0, random_state=42)

    train_small_multi = generate_multiclass(train_small, corpus, unseen).sample(frac=1.0, random_state=42)
    train_medium_multi = generate_multiclass(train_medium, corpus, unseen).sample(frac=1.0, random_state=42)
    train_large_multi = generate_multiclass(train_large, corpus, unseen).sample(frac=1.0, random_state=42)

    valid_small = generate_pairs(valid[0], corpus, valid_ccs[0]).sample(frac=1.0, random_state=42)
    valid_medium = generate_pairs(valid[1], corpus, valid_ccs[1]).sample(frac=1.0, random_state=42)
    valid_large = generate_pairs(valid[2], corpus, valid_ccs[2]).sample(frac=1.0, random_state=42)

    valid_small_multi = generate_multiclass(valid_small, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_medium_multi = generate_multiclass(valid_medium, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_large_multi = generate_multiclass(valid_large, corpus, unseen).sample(frac=1.0, random_state=42)
    
    valid_small_050_unseen = generate_pairs(valid_050_unseen[0], corpus, valid_ccs_050_unseen[0]).sample(frac=1.0, random_state=42)
    valid_medium_050_unseen = generate_pairs(valid_050_unseen[1], corpus, valid_ccs_050_unseen[1]).sample(frac=1.0, random_state=42)
    valid_large_050_unseen = generate_pairs(valid_050_unseen[2], corpus, valid_ccs_050_unseen[2]).sample(frac=1.0, random_state=42)

    valid_small_multi_050_unseen = generate_multiclass(valid_small_050_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_medium_multi_050_unseen = generate_multiclass(valid_medium_050_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_large_multi_050_unseen = generate_multiclass(valid_large_050_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    
    valid_small_100_unseen = generate_pairs(valid_050_unseen[0], corpus, valid_ccs_050_unseen[0]).sample(frac=1.0, random_state=42)
    valid_medium_100_unseen = generate_pairs(valid_050_unseen[1], corpus, valid_ccs_050_unseen[1]).sample(frac=1.0, random_state=42)
    valid_large_100_unseen = generate_pairs(valid_050_unseen[2], corpus, valid_ccs_050_unseen[2]).sample(frac=1.0, random_state=42)

    valid_small_multi_100_unseen = generate_multiclass(valid_small_100_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_medium_multi_100_unseen = generate_multiclass(valid_medium_100_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    valid_large_multi_100_unseen = generate_multiclass(valid_large_100_unseen, corpus, unseen).sample(frac=1.0, random_state=42)
    
    train_small.to_json(f'../data/derived/training-sets/products{name}_train_small.json.gz', lines=True, orient='records')
    train_medium.to_json(f'../data/derived/training-sets/products{name}_train_medium.json.gz', lines=True, orient='records')
    train_large.to_json(f'../data/derived/training-sets/products{name}_train_large.json.gz', lines=True, orient='records')
    
    train_small_multi.to_json(f'../data/derived/training-sets/productsmulti{name}_train_small.json.gz', lines=True, orient='records')
    train_medium_multi.to_json(f'../data/derived/training-sets/productsmulti{name}_train_medium.json.gz', lines=True, orient='records')
    train_large_multi.to_json(f'../data/derived/training-sets/productsmulti{name}_train_large.json.gz', lines=True, orient='records')

    valid_small.to_json(f'../data/derived/validation-sets/products{name}_valid_small.json.gz', lines=True, orient='records')
    valid_medium.to_json(f'../data/derived/validation-sets/products{name}_valid_medium.json.gz', lines=True, orient='records')
    valid_large.to_json(f'../data/derived/validation-sets/products{name}_valid_large.json.gz', lines=True, orient='records')
    
    valid_small_multi.to_json(f'../data/derived/validation-sets/productsmulti{name}_valid_small.json.gz', lines=True, orient='records')
    valid_medium_multi.to_json(f'../data/derived/validation-sets/productsmulti{name}_valid_medium.json.gz', lines=True, orient='records')
    valid_large_multi.to_json(f'../data/derived/validation-sets/productsmulti{name}_valid_large.json.gz', lines=True, orient='records')
    
    valid_small_050_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "050un")}_valid_small.json.gz', lines=True, orient='records')
    valid_medium_050_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "050un")}_valid_medium.json.gz', lines=True, orient='records')
    valid_large_050_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "050un")}_valid_large.json.gz', lines=True, orient='records')
    
    valid_small_100_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "100un")}_valid_small.json.gz', lines=True, orient='records')
    valid_medium_100_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "100un")}_valid_medium.json.gz', lines=True, orient='records')
    valid_large_100_unseen.to_json(f'../data/derived/validation-sets/products{name.replace("000un", "100un")}_valid_large.json.gz', lines=True, orient='records')
    
    valid_small_multi_050_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "050un")}_valid_small.json.gz', lines=True, orient='records')
    valid_medium_multi_050_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "050un")}_valid_medium.json.gz', lines=True, orient='records')
    valid_large_multi_050_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "050un")}_valid_large.json.gz', lines=True, orient='records')
    
    valid_small_multi_100_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "100un")}_valid_small.json.gz', lines=True, orient='records')
    valid_medium_multi_100_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "100un")}_valid_medium.json.gz', lines=True, orient='records')
    valid_large_multi_100_unseen.to_json(f'../data/derived/validation-sets/productsmulti{name.replace("000un", "100un")}_valid_large.json.gz', lines=True, orient='records')
       
    test_000_unseen.to_json(f'../data/derived/gold-standards/products{name}_gs.json.gz', lines=True, orient='records')
    test_multi_000_unseen.to_json(f'../data/derived/gold-standards/productsmulti{name}_gs.json.gz', lines=True, orient='records')
    
    test_050_unseen.to_json(f'../data/derived/gold-standards/products{name.replace("000un", "050un")}_gs.json.gz', lines=True, orient='records')
    test_multi_050_unseen.to_json(f'../data/derived/gold-standards/productsmulti{name.replace("000un", "050un")}_gs.json.gz', lines=True, orient='records')
    
    test_100_unseen.to_json(f'../data/derived/gold-standards/products{name.replace("000un", "100un")}_gs.json.gz', lines=True, orient='records')
    test_multi_100_unseen.to_json(f'../data/derived/gold-standards/productsmulti{name.replace("000un", "100un")}_gs.json.gz', lines=True, orient='records')

