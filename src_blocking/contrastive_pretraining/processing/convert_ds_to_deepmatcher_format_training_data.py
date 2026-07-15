import logging
import os
import random

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option('--dataset')
@click.option('--testset')
@click.option('--size', default='large')
@click.option('--modify_data', default=False)
@click.option('--datadir', default=os.environ.get("DATA_DIR"))





def convert_table_to_query_table(dataset, testset, size, modify_data, datadir):
    def reconstruct_pairs(split_df, tableA, tableB):

        # Join left table
        df = split_df.merge(
            tableA,
            left_on="ltable_id",
            right_on="id_left"
        )


        # Join right table
        df = df.merge(
            tableB,
            left_on="rtable_id",
            right_on="id_right"
        )
        df = df.drop(columns=["ltable_id", "rtable_id"])
        df = df.rename(columns={
            "product_id_left_y": "product_id_left",
            "product_id_right_y": "product_id_right"
        })
        df = df.drop(columns=[
            "product_id_left_x",
            "product_id_right_x"
        ])
        return df
    
    random.seed(42)
    base_path = f"{datadir}/{size}"

    tableA = pd.read_csv(f"{base_path}/tableA.csv")
    #add in all column names the suffix _left
    tableA = tableA.add_suffix("_left")
    tableB = pd.read_csv(f"{base_path}/tableB.csv")
    #add in all column names the suffix _right
    tableB = tableB.add_suffix("_right")

    ds_sets = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        df = pd.read_csv(f"{base_path}/{split}.csv")

        df_split = reconstruct_pairs(df, tableA, tableB)

        ds_sets[split] = df_split.sort_values(by=['label'], ascending=False)

    print("DATA_DIR:", datadir)
    print("Dataset:", dataset)
    print("Testset:", testset)
    print("Columns:", ds_sets['test'].columns.tolist())
    print("Train size:", len(ds_sets['train']))

    if modify_data:
        # Determine unique cluster ids - all clusters appear left and right
        product_ids = [ds_sets[split]['product_id_left'].unique() for split in ds_sets]
        product_ids = set([product_id for product_id_list in product_ids for product_id in product_id_list])
        not_test_product_ids = [ds_sets[split]['product_id_left'].unique() for split in ds_sets if split in ['train', 'valid']]
        not_test_product_ids = set([product_id for product_id_list in not_test_product_ids for product_id in product_id_list])

        # Make sure that seen records are really seen during training! --> One pair from the cluster
        product_ids_to_record_ids = {}

        unique_ids = set()
        for split in ds_sets:
            print(len(ds_sets[split]))
            unique_ids.update(ds_sets[split]['id_left'].unique())

        print(len(unique_ids))
        print(ds_sets['test'].columns)

        swap_columns = ['id', 'brand', 'name', 'desc', 'price', 'product_id']

        # Determine left offer ids per cluster id
        for product_id in tqdm(product_ids):
            # Get Offer ids for train & valid
            offer_ids = [ds_sets[split].loc[(ds_sets[split]['product_id_left'] == product_id)]['id_left'].unique()
                          for split in ds_sets]
            offer_ids = list(set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list]))
            left_offer_id = offer_ids.pop()  # Leading offer id

            # train_valid_offer_ids = [ds_sets[split].loc[(ds_sets[split]['product_id_left'] == product_id)]['id_left'].unique()
            #              for split in ['train', 'valid']]
            # train_valid_offer_ids = list(set([offer_id for offer_id_list in train_valid_offer_ids
            #                                   for offer_id in offer_id_list]))
            # test_offer_ids = list(ds_sets['test'].loc[(ds_sets['test']['product_id_left'] == product_id)]['id_left'].unique())
            #
            # if len(train_valid_offer_ids) > 0:
            #     # Determine lead offer id from train/valid
            #     train_valid_offer_ids.sort()
            #     left_offer_id = train_valid_offer_ids.pop() # Leading offer id
            # else:
            #     # Determine lead offer id from test if the cluster is only present in the test set
            #     test_offer_ids.sort()
            #     left_offer_id = test_offer_ids.pop()
            #
            # offer_ids = set(train_valid_offer_ids + test_offer_ids)

            #product_ids_to_record_ids[product_id] = {'left_offer_id': left_offer_id, 'offer_ids': list(offer_ids)}

            for split in ds_sets:
                # # Swap leading offer ids if they appear on the right hand site of a pair
                # swap_rows = ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id]
                # for column in swap_columns:
                #     swap_rows['{}_temp'.format(column)] = swap_rows['{}_right'.format(column)]
                #     swap_rows['{}_right'.format(column)] = swap_rows['{}_left'.format(column)]
                #     swap_rows['{}_left'.format(column)] = swap_rows['{}_temp'.format(column)]

                # Replace left hand side offer ids by leading offer id
                for offer_id in offer_ids:
                    ds_sets[split].loc[ds_sets[split]['id_left'] == offer_id, 'id_left'] = left_offer_id

                # Replace right hand side leading offer ids with random offer id from the same cluster
                ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(list(offer_ids))
                # if left_offer_id in ds_sets[split]['id_right']:
                #     if split == 'test':
                #         ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(test_offer_ids)
                #     else:
                #         ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(
                #             train_valid_offer_ids)
                #ds_sets[split] = ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id]

                ds_sets[split] = ds_sets[split].drop_duplicates()

                # Delete all other offer ids from the left hand sight
                # swap_rows = ds_sets[split].loc[ds_sets[split]['id_left'].isin(offer_ids)]
                # for column in swap_columns:
                #     swap_rows['{}_temp'.format(column)] = swap_rows['{}_right'.format(column)]
                #     swap_rows['{}_right'.format(column)] = swap_rows['{}_left'.format(column)]
                #     swap_rows['{}_left'.format(column)] = swap_rows['{}_temp'.format(column)]
                #ds_sets[split] = ds_sets[split].loc[~ds_sets[split]['id_left'].isin(offer_ids)]
                #ds_sets[split] = pd.concat([ds_sets[split], swap_rows])

        # Remove pairs where both records are supposed to be in the query table
        unique_left_ids = set()
        for split in ds_sets:
            #print(len(ds_sets[split]))
            unique_left_ids.update(ds_sets[split]['id_left'].unique())

        for split in ds_sets:
            ds_sets[split] = ds_sets[split].loc[~ds_sets[split]['id_right'].isin(unique_left_ids)]


        #unique_ids = set()
        for product_id in tqdm(product_ids):
            offer_ids = [ds_sets[split].loc[(ds_sets[split]['product_id_left'] == product_id) & (
                        ds_sets[split]['product_id_right'] == product_id)]['id_left'].unique() for split in ds_sets]
            offer_ids = set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list])

        for split in splits:
            print(len(ds_sets[split]))
            print(len(ds_sets[split].loc[ds_sets[split]['label'] == 1]))

    table_A_records = []
    table_B_records = []
    split_info = {'train': [], 'valid': [], 'test': []}
    print(ds_sets['test'].columns)
    for split in splits:
        for index, row in ds_sets[split].iterrows():
            left_record = {'id': row['id_left'], 'brand': row['brand_left'], 'name': row['name_left'],
                           'desc': row['desc_left'], 'price': row['price_left'],
                           'product_id': row['product_id_left']}
            table_A_records.append(left_record)

            right_record = {'id': row['id_right'],'brand': row['brand_right'], 'name': row['name_right'],
                           'desc': row['desc_right'],'price': row['price_right'],
                           'product_id': row['product_id_right']}
            table_B_records.append(right_record)

            matching_info = {'ltable_id': row['id_left'], 'rtable_id': row['id_right'], 'label': row['label'], 'product_id': row['product_id_left']}
            split_info[split].append(matching_info)

    # Create Data Frames
    df_table_a = pd.DataFrame(table_A_records)
    df_table_b = pd.DataFrame(table_B_records)

    # #Add matches to train/ validation to make sure that each cluster is seen at least once.
    # for product_id in not_test_product_ids:
    #     product_id_to_record_ids = product_ids_to_record_ids[product_id]
    #     for _ in range(2):
    #         if len(offer_ids) > 0:
    #             matching_info = {'ltable_id': product_id_to_record_ids['left_offer_id'],
    #                              'rtable_id': product_id_to_record_ids['offer_ids'].pop(), 'label': 1, 'product_id': product_id}
    #             if matching_info not in split_info['train'] and matching_info not in split_info['valid'] \
    #                 and matching_info not in split_info['test']:
    #                 if random.randint(1, 4) == 1:
    #                     split_info['valid'].append(matching_info)
    #                 else:
    #                     split_info['train'].append(matching_info)

    df_train = pd.DataFrame(split_info['train']).drop(columns=['product_id'])
    df_valid = pd.DataFrame(split_info['valid']).drop(columns=['product_id'])
    df_test = pd.DataFrame(split_info['test']).drop(columns=['product_id'])

    # Drop duplicates from data tables
    df_table_a = df_table_a.drop_duplicates(subset=['id'])
    df_table_b = df_table_b.drop_duplicates(subset=['id'])

    # Save Data Frames
    path = '{}/products80cc20rnd_{}'.format(os.environ['DATA_DIR'], testset)

    if not os.path.exists(path):
        os.makedirs(path)

    df_table_a = df_table_a.set_index('id')
    df_table_a.to_csv(path_or_buf='{}/tableA.csv'.format(path), sep=',')

    df_table_b = df_table_b.set_index('id')
    df_table_b.to_csv(path_or_buf='{}/tableB.csv'.format(path), sep=',')

    df_train.to_csv(path_or_buf='{}/train.csv'.format(path), sep=',', index=False)
    df_valid.to_csv(path_or_buf='{}/valid.csv'.format(path), sep=',', index=False)
    df_test.to_csv(path_or_buf='{}/test.csv'.format(path), sep=',', index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_query_table()