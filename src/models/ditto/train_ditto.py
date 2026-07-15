import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

import nltk
#nltk.download('stopwords')

from codecarbon import OfflineEmissionsTracker
import time, os, json, pandas as pd

def run_with_tracking(job_name, func, *args,
                      electricity_price_eur_per_kwh=0.30,
                      **kwargs):

    os.makedirs("data/efficiency_tracker/ditto", exist_ok=True)
    json_path = f"data/efficiency_tracker/ditto/{job_name}.json"
    csv_path = f"data/efficiency_tracker/ditto/{job_name}.csv"

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_file=csv_path
    )
    # ---- GPU MEMORY RESET (BEFORE TRAINING) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    tracker.start()

    func(*args, **kwargs)

    tracker.stop()
    runtime_sec = time.time() - start_time

    # ---- PEAK GPU MEMORY (AFTER TRAINING) ----
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        max_memory_mb = None

    emission_df = pd.read_csv(csv_path)
    energy_kwh = emission_df["energy_consumed"].iloc[-1]
    emissions_kg = emission_df["emissions"].iloc[-1]
    energy_cost_eur = energy_kwh * electricity_price_eur_per_kwh

    record = {
        "job_name": job_name,
        "runtime_sec": round(runtime_sec, 3),
        "max_memory_mb": None if max_memory_mb is None else round(max_memory_mb, 3),
        "energy_kwh": round(energy_kwh, 6),
        "emissions_kg": round(emissions_kg, 6),
        "energy_cost_eur": round(energy_cost_eur, 4),
    }

    data = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)

    data.append(record)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    mem_str = "CPU" if max_memory_mb is None else f"{max_memory_mb:.2f} MB"
    print(f"Runtime: {runtime_sec:.2f}s | Max Memory: {mem_str} MB")
    print(f"Energy: {energy_kwh:.6f} kWh | CO₂: {emissions_kg:.6f} kg | Total Cost: {energy_cost_eur:.4f} €")
    print(f"Results appended to: {json_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="src/models/ditto/checkpoints/")
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_id=%d_adjusted_testset' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('src/models/ditto/configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    if (task in configs):
        config = configs[task]
    else:
        raise KeyboardInterrupt

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    testset050 = config['testset050']
    testset100 = config['testset100']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)
        testset050 = summarizer.transform_file(testset050, max_len=hp.max_len)
        testset100 = summarizer.transform_file(testset100, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)
        testset050 = injector.transform_file(testset050)
        testset100 = injector.transform_file(testset100)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)
    test_dataset050 = DittoDataset(testset050, lm=hp.lm)
    test_dataset100 = DittoDataset(testset100, lm=hp.lm)

    job_name = f"{run_tag}"
    path = "src/models/ditto/output/"
    run_with_tracking(
        job_name=job_name,
        func=train,
        trainset=train_dataset,
        validset=valid_dataset,
        testset=test_dataset,
        testset050=test_dataset050,
        testset100=test_dataset100,
        run_tag=run_tag,
        hp=hp,
        path = path
    )