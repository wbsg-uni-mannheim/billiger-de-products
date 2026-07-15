import logging
import os
from collections import defaultdict

import click
import pandas as pd

from src_blocking.model.evidence import RetrievalEvidence
from src_blocking.model.querytable import RetrievalQueryTable

import csv
from collections import defaultdict

@click.command()
@click.option('--dataset')
@click.option('--table_name', default='tableA')
def convert_table_to_query_table(dataset, table_name):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param dataset string org class represents the dataset name"""

    switched = False if table_name == 'tableA' else True  # Switched table A and B
    if switched:
        logging.info('Switched table A and B')

    path_to_table_a = '{}/{}/{}.csv'.format(
        os.environ['DATA_DIR'], dataset, table_name)
    path_to_test_set = '{}/{}/test.csv'.format(os.environ['DATA_DIR'], dataset)
    path_to_train_set = '{}/{}/train.csv'.format(os.environ['DATA_DIR'], dataset)
    path_to_valid_set = '{}/{}/valid.csv'.format(os.environ['DATA_DIR'], dataset)

    print(os.environ['DATA_DIR']) #TODO remove
    # Add all records as evidences to query table
    test_record_dict = defaultdict(list)
        # Add all records as evidences to query table
    test_record_dict = defaultdict(list)

    def flatten_value(x):
        """Convert list-like, quoted, or strange values into flat strings."""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, (list, tuple)):
            return " ".join(map(str, x))
        return str(x)

    for split in ['train', 'valid', 'test']:
        path = '{}/{}/{}.csv'.format(os.environ['DATA_DIR'], dataset, split)

        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row_num, line_values in enumerate(reader):

                # Skip header
                if row_num == 0 and line_values and line_values[0] == "ltable_id":
                    continue

                # Guarantee at least 3 columns so we don't crash
                if len(line_values) < 3:
                    logging.warning(f"[WARN] Malformed row {row_num} in {split}: {line_values}")
                    # Pad so indexes exist
                    while len(line_values) < 3:
                        line_values.append("")

                # Flatten columns (this matches your __getitem__ behavior)
                col0 = flatten_value(line_values[0])
                col1 = flatten_value(line_values[1])
                col2 = flatten_value(line_values[2])

                if table_name == 'tableA':
                    test_record_dict[col0].append({
                        'row_id': col1,
                        'label': col2,
                        'split': split
                    })

                elif table_name == 'tableB':
                    test_record_dict[col1].append({
                        'row_id': col0,
                        'label': col2,
                        'split': split
                    })


    # Extract seen records
    seen_evidences_records = set()
    seen_entity_records = set()
    #seen_pair = []
    for path in [path_to_train_set, path_to_valid_set]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_values = line.split(',')
                if line_values[0] == 'ltable_id':
                    continue
                if line_values[2] == '0':
                    if table_name == 'tableA':
                        # Take seen record from table B
                        #seen_pair.append(f'{line_values[0]}#{line_values[1]}')
                        seen_evidences_records.add(line_values[1])
                        seen_entity_records.add(line_values[0])
                    elif table_name == 'tableB':
                        # Take seen evidence record from table A
                        #seen_pair.append(f'{line_values[1]}#{line_values[0]}')
                        seen_evidences_records.add(line_values[0])
                        seen_entity_records.add(line_values[1])



    # Build Query Table
    query_table_ids = {
        'small': 20000,
        'medium': 30000,
        'large': 40000,
    }


    qt_id = query_table_ids[dataset]
    assembling_strategy = 'Test {} of data set {}'.format(table_name, dataset)
    gt_table = dataset


    verified_evidences = []
    table = []
    evidence_id = 1

    df_table = pd.read_csv(path_to_table_a).fillna('')
    context_attributes = df_table.columns.to_list()[1:]
    context_attributes = [c.lower() for c in context_attributes]  # normalize
    df_table.columns = ['id'] + context_attributes

    seen_counter = {'both_seen': 0, 'left_seen': 0, 'right_seen': 0, 'none_seen': 0}

    for index, row in df_table.iterrows():
        if len(table) >= 100:
            # Check data sets into query tables with 100 records
            query_table = RetrievalQueryTable(qt_id, 'retrieval', assembling_strategy,
                                              gt_table, dataset,
                                              context_attributes,  # Exclude id
                                              table, verified_evidences)
            query_table.switched = switched
            query_table.save(with_evidence_context=False)

            # Initialize variables for new query table
            verified_evidences = []
            table = []
            evidence_id = 1
            qt_id += 1

        entity = row.to_dict()
        entity = dict((k.lower(), v) for k, v in entity.items())

        entity['entityId'] = entity['id']
        del entity['id']

        #if 'products' in dataset:
        del entity['product_id']
            #context_attributes = list(map(lambda x: x.replace('Pant', 'Ishan'), context_attributes))

        #added_pairs = []
        for reference in test_record_dict[str(entity['entityId'])]:
            #pair_id = '{}#{}'.format(entity['entityId'], reference['row_id'])
            #if pair_id not in added_pairs:
            table_identifier = '{}_{}.json.gz'.format(dataset, 'tableA' if table_name == 'tableB' else 'tableB').lower()
            evidence = RetrievalEvidence(evidence_id, qt_id, entity['entityId'],
                                         table_identifier, reference['row_id'], None, reference['split'])
            evidence.scale = int(reference['label'])
            evidence.signal = int(reference['label']) == 1

            if str(entity['entityId']) in seen_entity_records and reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['both_seen'] += 1
            elif str(entity['entityId']) in seen_entity_records:
                evidence.seen_training = 'left_seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['left_seen'] += 1
            elif reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'right_seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['right_seen'] += 1
            else:
                evidence.seen_training = 'unseen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['none_seen'] += 1

            # = str(entity['entityId']) in seen_entity_records or reference['row_id'] in seen_evidences_records

            #reference['row_id'] in seen_evidences_records or entity['entityId'] in seen_entity_records

            verified_evidences.append(evidence)
                #added_pairs.append(pair_id)
            evidence_id += 1

        #if len(test_record_dict[str(entity['entityId'])]) > 0:
        # Add all entities of table a to query table
        table.append(entity)

    # Save final query table
    query_table = RetrievalQueryTable(qt_id, 'retrieval', assembling_strategy,
                                      gt_table, dataset,
                                      context_attributes,  # Exclude id
                                      table, verified_evidences)
    query_table.switched = switched
    query_table.save(with_evidence_context=False)
    logging.info('Converted {} of {} to query table'.format(table_name, dataset))

    print(seen_counter)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_query_table()
