#!/usr/bin/env python3

import logging
import time
from datetime import datetime

import numpy as np
from torch.multiprocessing import Pool, set_start_method
from random import randrange
import random

import click
import torch
import yaml

from src_blocking.evaluation.aggregate_results import aggregate_results, save_aggregated_result
from src_blocking.evaluation.evaluate_query_tables import evaluate_query_table
from src_blocking.model.querytable import load_query_table_from_file, get_gt_tables, get_query_table_paths
from src_blocking.strategy.ranking.similarity.similarity_re_ranking_factory import select_similarity_re_ranker
from src_blocking.strategy.retrieval.retrieval_strategy_factory import select_retrieval_strategy
from src_blocking.strategy.pipeline_building import build_pipelines_from_configuration, validate_configuration

from src_blocking.evaluation.save_blocking_pairs import save_blocking_pairs

from codecarbon import OfflineEmissionsTracker
import os
import json
import pandas as pd

def run_with_tracking(job_name, func, *args, gpt_usage=False, electricity_price_eur_per_kwh=0.30, **kwargs):

    os.makedirs("data/efficiency_tracker/blocking", exist_ok=True)
    json_path = f"data/efficiency_tracker/blocking/{job_name}.json"
    csv_path = f"data/efficiency_tracker/blocking/{job_name}.csv"

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_file=csv_path
    )
    # ---- GPU MEMORY RESET (BEFORE TRAINING) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    tracker.start()


    results, execution_times = func(*args, **kwargs)  

    tracker.stop()
    end_time = time.time()
    runtime_sec = time.time() - start_time

    # ---- PEAK GPU MEMORY (AFTER TRAINING) ----
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        max_memory_mb = None

    # Calculate energy and costs
    emission_df = pd.read_csv(csv_path)
    energy_kwh = emission_df["energy_consumed"].iloc[-1]
    emissions_kg = emission_df["emissions"].iloc[-1]
    energy_cost_eur = energy_kwh * electricity_price_eur_per_kwh

    if not gpt_usage:
        total_cost_eur = energy_cost_eur
    else:
        cost = pd.read_csv("dataset_quality_test.csv")
        gpt_cost = cost["Costs"].sum()

    total_cost_eur = energy_cost_eur + gpt_cost

    # Log result
    record = {
        "job_name": job_name,
        "runtime_sec": round(runtime_sec, 3),
        "max_memory_mb": None if max_memory_mb is None else round(max_memory_mb, 3),
        "energy_kwh": round(energy_kwh, 6),
        "emissions_kg": round(emissions_kg, 6),
        "energy_cost_eur": round(energy_cost_eur, 4),
        "total_cost_eur": round(total_cost_eur, 4),
    }

    # Append or create JSON file
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    mem_str = "CPU" if max_memory_mb is None else f"{max_memory_mb:.2f} MB"
    print(f"Runtime: {runtime_sec:.2f}s | Max Memory: {mem_str} MB")
    print(f"Energy: {energy_kwh:.6f} kWh | CO₂: {emissions_kg:.6f} kg | Total Cost: {total_cost_eur:.4f} €")
    print(f"Results appended to: {json_path}")

    return results, execution_times

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@click.command()
@click.option('--path_to_config')
@click.option('--worker', type=int, default=0)
def run_experiments_from_configuration(path_to_config, worker):
    logger = logging.getLogger()

    set_seed(42)
    # Load yaml configuration
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validate_configuration(config)

    config_name = path_to_config.split('/')[-1].replace('.yml','')

    context_attributes = config['query-tables']['context-attributes']
    experiment_type = config['general']['experiment-type']

    # Load query tables
    dataset = config['query-tables']['dataset']
    switched = config['query-tables']['switched'] if 'switched' in config['query-tables'] else False
    logger.info('Switched: {}'.format(switched))
    query_table_paths = []
    if type(config['query-tables']['path-to-query-table']) is str:
        # Run for single query table
        query_table_paths.append(config['query-tables']['path-to-query-table']) # query_table_paths must be an array
    elif config['query-tables']['gt-table'] is None: # TODO changed by Ksenia from is not None to is None
        # Run on query tables for gt table
        query_table_paths.extend(get_query_table_paths(config['general']['experiment-type'],
                                                  config['query-tables']['dataset'],
                                                  config['query-tables']['gt-table'], switched=switched))
    else:
        # Run on all query tables of the dataset
        for gt_table in get_gt_tables(config['general']['experiment-type'], dataset):
            query_table_paths.extend(get_query_table_paths(config['general']['experiment-type'], dataset,
                                                           gt_table, switched=switched))

    string_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if config['general']['k'] is not None:
        if isinstance(config['general']['k'], list):
            k_range = config['general']['k']
        else:
            k_range = [config['general']['k']]
    elif config['general']['k_range'] is not None:
        k_range = range(config['general']['k_range'][0], config['general']['k_range'][1] + 1)

    logger.info(f'Will test the following values of k: {k_range}')

    save_results_with_evidences = config['general']['save_results_with_evidences']
    clusters = config['general']['clusters']
    #os.environ["ES_INSTANCE"] = config['general']['es_instance']

    pool = None
    async_results = None
    if worker > 0:
        pool = Pool(worker)
        async_results = []

    file_name = 'results_{}_{}.json'.format(string_timestamp, config_name)

    # Build pipelines from yaml configuration
    pipelines = build_pipelines_from_configuration(config)

    # Start run experiments by combining pipelines and query tables
    for k in k_range:
        logger.info('Start running experiments for k={}'.format(k))
        for pipeline in pipelines:
            logger.info('Amount of pipelines to run: {}'.format(len(pipelines)))
            retrieval_strategy = pipeline['retrieval_strategy']
            similarity_re_ranking_strategy = pipeline['similarity_re_ranking_strategy']
            source_re_ranking_strategy = pipeline['source_re_ranking_strategy']
            voting_strategies = pipeline['voting_strategies']
            logger.info('Amount of workers: {}'.format(worker))
            if worker == 0:
                # -------------------------------------------------------
                # Efficiency-tracked wrapper (PER k!)
                # -------------------------------------------------------
                def experiment_wrapper():
                    return run_experiments(
                        experiment_type,
                        retrieval_strategy,
                        similarity_re_ranking_strategy,
                        source_re_ranking_strategy,
                        voting_strategies,
                        query_table_paths,
                        dataset,
                        k,
                        context_attributes,
                        clusters=clusters,
                        switched=switched,
                    )

                job_name = f"scblock_{dataset}_k{k}"

                results, execution_times = run_with_tracking(
                    job_name=job_name,
                    func=experiment_wrapper,
                )
                if results is not None:
                    for result in results:
                        result.save_result(file_name, save_results_with_evidences)

                    aggregated_result = aggregate_results(results, k, execution_times)
                    save_aggregated_result(aggregated_result, file_name)

            elif worker > 0:

                async_results.append(pool.apply_async(run_experiments, (experiment_type, retrieval_strategy,
                                                                        similarity_re_ranking_strategy,
                                                                        source_re_ranking_strategy, voting_strategies,
                                                                        query_table_paths, dataset, k,
                                                                        context_attributes, clusters, switched,)))

        if worker > 0:
            logger.info('Waiting for all experiments to finish!')

            while len(async_results) > 0:
                logger.info('Number of chunks: {}'.format(len(async_results)))
                #check elasticsearch connection 
                time.sleep(5)
                async_results = collect_results_of_finished_experiments(async_results, file_name, k,
                                                                        save_results_with_evidences, True)

    if worker > 0:
        pool.close()
    logger.info('Finished running experiments!')
    logger.info('Results are saved to {}'.format(file_name))


def run_experiments(experiment_type, retrieval_str_conf, similarity_re_ranking_str_conf, source_re_ranking_str_conf,
                    voting_strategies, query_table_paths, dataset, evidence_count, context_attributes=None, clusters=False, switched=False):
    """Run Pipeline on query tables"""

    time.sleep(randrange(30))
    logger = logging.getLogger()
    # Initialize strategy
    retrieval_strategy = select_retrieval_strategy(retrieval_str_conf, dataset, clusters, switched)
    similarity_re_ranker = select_similarity_re_ranker(similarity_re_ranking_str_conf, dataset,
                                                       context_attributes)
    #source_re_ranker = select_source_re_ranker(source_re_ranking_str_conf, dataset)
    source_re_ranker = None # Exclude source re-ranking for now
    logger.info('Run experiments on {} query tables'.format(len(query_table_paths)))
    results = []
    execution_times = []

    materialized_pairs = []
    for query_table_path in query_table_paths:

        query_table = load_query_table_from_file(query_table_path)
        # FIX context attributes
        if experiment_type == 'augmentation' and context_attributes is not None:
            if query_table.target_attribute in context_attributes:
                continue
            # Run experiments only on a subset of context attributes
            removable_attributes = [attr for attr in query_table.context_attributes
                                    if attr not in context_attributes and attr != 'name']
            for attr in removable_attributes:
                query_table.remove_context_attribute(attr)
        
        logger.warning(f"[{query_table_path}] start retrieval k={evidence_count}") #TODO
        query_table.retrieved_evidences, execution_times_per_run = retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                                     similarity_re_ranker, source_re_ranker)
        logger.warning(f"[{query_table_path}] done retrieval: {len(query_table.retrieved_evidences)}") #TODO

        materialized_pairs.extend(query_table.materialize_pairs())

        logger.warning(f"[{query_table_path}] done materialize_pairs") #TODO
        pairs = query_table.materialize_pairs()
        logger.warning(f"[{query_table_path}] done evaluate_query_table: {len(pairs)}") #TODO

        save_blocking_pairs(
            pairs=pairs,
            dataset=dataset,
            k=evidence_count,
            split="test",  # oder query_table.split, falls vorhanden
            out_dir=f"{os.environ['DATA_DIR']}/results/{dataset}/blocking_pairs"
        )

        execution_times.append(execution_times_per_run)

        if retrieval_str_conf['name'] == 'generate_entity':
            k_intervals = [1]
        else:
            k_intervals = [evidence_count]

        for voting_str_conf in voting_strategies:
            split = None if similarity_re_ranker is None else 'test'
            print(f"[{query_table_path}] start evaluate_query_table") #TODO
            new_results = evaluate_query_table(query_table, experiment_type, retrieval_strategy, similarity_re_ranker,
                                               source_re_ranker, k_intervals, voting_str_conf['name'], split=split,
                                               collect_result_context=True)
            print(f"[{query_table_path}] done evaluate_query_table: {len(new_results)}") #TODO
            results.extend(new_results)

    aggregated_execution_times = {key: sum([execution_time[key] for execution_time in execution_times])
                                  for key in execution_times[0]}

    print('Finished running experiments on subset of query tables!')

    return results, aggregated_execution_times


def retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                     similarity_re_ranker, source_re_ranker, entity_id=None):
    execution_times = {}
    start_time = time.time()
    # Run retrieval strategy
    evidences = retrieval_strategy.retrieve_evidence(query_table, evidence_count, entity_id)
    retrieval_time = time.time()
    execution_times['retrieval_time'] = retrieval_time - start_time

    # Filter evidences by ground truth tables
    evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(evidences)

    # Run re-ranker
    if similarity_re_ranker is not None:
        # Re-rank evidences by cross encoder - to-do: Does it make sense to track both bi encoder and reranker?
        evidences = similarity_re_ranker.re_rank_evidences(query_table, evidences)
        similarity_re_ranking_time = time.time()
        execution_times['sim_reanker_time'] = similarity_re_ranking_time - retrieval_time

    if source_re_ranker is not None:
        # Re-rank evidences by source information
        evidences = source_re_ranker.re_rank_evidences(query_table, evidences)
        source_re_ranking_time = time.time()
        if similarity_re_ranker is not None:
            execution_times['source_reranker_time'] = source_re_ranking_time - similarity_re_ranking_time
        else:
            execution_times['source_reranker_time'] = source_re_ranking_time - retrieval_time

    execution_times['complete_execution_time'] = time.time() - start_time

    return evidences, execution_times


def collect_results_of_finished_experiments(async_results, file_name, evidence_count, with_evidences=True, with_extended_results=False):
    """Collect results and write them to file"""
    logger = logging.getLogger()
    collected_results = []
    for async_result in async_results:
        if async_result.ready():
            results, execution_times = async_result.get()
            collected_results.append(async_result)

            # Save query table to file
            if results is not None:
                logger.info('Will collect {} results now!'.format(len(results)))
                if with_extended_results:
                    for result in results:
                        result.save_result(file_name, with_evidences)

                #for i in range(1, 11):
                aggregated_result = aggregate_results(results, evidence_count, execution_times)
                save_aggregated_result(aggregated_result, file_name)

    # Remove collected results from list of results
    async_results = [async_result for async_result in async_results if async_result not in collected_results]

    return async_results


def run_strategy_to_retrieve_evidence(query_table_id, schema_org_class, experiment_type, retrieval_str_conf,
                                      similarity_re_ranking_str_conf, source_re_ranking_str_conf, entity_id=None):
    # TO-DO: UPDATE SO THAT THE ANNOTATION TOOL CONTINUES TO WORK!
    # Initialize Table Augmentation Strategy
    evidence_count = 30  # Deliver 20 evidence records for now

    #To-Do: Does it make sense to set clusters always to true?
    retrieval_strategy = select_retrieval_strategy(retrieval_str_conf, schema_org_class, clusters=True)
    similarity_re_ranker = select_similarity_re_ranker(similarity_re_ranking_str_conf, schema_org_class)
    #source_re_ranker = select_source_re_ranker(source_re_ranking_str_conf, schema_org_class)
    source_re_ranker = None # Exclude source re-ranker for now

    query_table = None
    context_attributes = ['name', 'addresslocality']

    for gt_table in get_gt_tables(experiment_type, schema_org_class):
        for query_table_path in get_query_table_paths('retrieval', schema_org_class, gt_table):
            if query_table_path.endswith('_{}.json'.format(query_table_id)):
                query_table = load_query_table_from_file(query_table_path)
                # Run experiments only on a subset of context attributes
                removable_attributes = [attr for attr in query_table.context_attributes
                                        if attr not in context_attributes and attr != 'name']
                for attr in removable_attributes:
                    query_table.remove_context_attribute(attr)

    evidences = retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                                 similarity_re_ranker, source_re_ranker, entity_id=entity_id)

    # Return evidence --> (Filter for single entity)
    requested_evidence = [evidence for evidence in evidences
                          if evidence.entity_id == entity_id]

    return requested_evidence[:evidence_count]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    set_start_method('spawn')
    run_experiments_from_configuration()
