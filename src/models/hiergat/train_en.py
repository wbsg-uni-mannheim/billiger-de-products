import argparse
import json
import os

from model.eval import eval_on_task
from model.dataset import Dataset, get_tokenizer
from model.model import TranHGAT

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.utils import data
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import time

import sys
import random

from codecarbon import OfflineEmissionsTracker
import time, os, json, pandas as pd

def creat_pred_csv(path_predictions, best_test_preds):
    os.makedirs(os.path.dirname(path_predictions), exist_ok=True)
    with open(path_predictions, "w") as f:
        f.write("label,prediction\n")
        for i in range(len(best_test_preds['labels'])):
            f.write(
                f"{best_test_preds['labels'][i]},"
                f"{best_test_preds['preds'][i]}\n"
            )

def run_with_tracking(job_name, func, *args,
                      electricity_price_eur_per_kwh=0.30,
                      **kwargs):

    os.makedirs("data/efficiency_tracker/hiergat_en", exist_ok=True)
    json_path = f"data/efficiency_tracker/hiergat_en/{job_name}.json"
    csv_path = f"data/efficiency_tracker/hiergat_en/{job_name}.csv"

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


def train(model, train_set, optimizer, scheduler=None, batch_size=32):
    iterator = data.DataLoader(dataset=train_set, batch_size=batch_size,
                               shuffle=True, num_workers=1, collate_fn=Dataset.pad)
    classifier_criterion = nn.CrossEntropyLoss()

    model.train()
    for i, batch in enumerate(iterator):
        # for monitoring
        _, xs, y, _, masks = batch
        _y = y

        # forward
        optimizer.zero_grad()
        logits, y, _ = model(xs, y, masks)

        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = classifier_criterion(logits, y)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
            del loss


def initialize_and_train(trainset, validset, testset, testset050, testset100, attr_num, args, run_tag):
    padder = Dataset.pad
    valid_iter = data.DataLoader(dataset=validset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, collate_fn=padder)
    test_iter050 = data.DataLoader(dataset=testset050, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, collate_fn=padder)
    test_iter100 = data.DataLoader(dataset=testset100, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, collate_fn=padder)

    # initialize model
    # set seeds
    seed = args.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TranHGAT(attr_num, device, args.finetuning, lm=args.lm, lm_path=args.lm_path)
    if device == 'cpu':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # learning rate scheduler
    num_steps = (len(trainset) // args.batch_size) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # create logging directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    writer = SummaryWriter(log_dir=args.logdir)

    # start training
    best_dev_f1 = best_test_f1 = best_test_f1_050 = best_test_f1_100 = 0.0
    epoch = 1
    no_improvement_count = 0
    best_test_preds = None
    best_test_preds_050 = None
    best_test_preds_100 = None
    
    while ((epoch <= args.n_epochs) and (no_improvement_count <= 10)):
        start = time.time()
        train(model, trainset, optimizer, scheduler=scheduler,
              batch_size=args.batch_size)
        print("train time: ", time.time()-start)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1, test_preds, test_f1_050, test_preds_050, test_f1_100, test_preds_100 = eval_on_task(epoch, model, valid_iter, test_iter, test_iter050, test_iter100,
                                       writer, run_tag, return_preds=True)

        if dev_f1 > 1e-6:
            epoch += 1

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_test_f1_050 = test_f1_050
                best_test_f1_100 = test_f1_100
                best_test_preds = test_preds
                best_test_preds_050 = test_preds_050
                best_test_preds_100 = test_preds_100
            if (epoch == args.n_epochs):
                path = os.getcwd()
                path = path + '/src/models/hiergat/output_en/' + str(run_tag) + '.txt'
                os.makedirs(os.path.dirname(path), exist_ok=True)

                dict = {'best_test_f1': best_test_f1, 'best_test_f1_050': best_test_f1_050, 'best_test_f1_100': best_test_f1_100}

                with open(path, "a+") as f:
                    f.write(repr(dict) + '\n')
  
                path_predictions = "src/models/hiergat/output_en/prediction/" + str(run_tag) + '_predictions.csv'
                os.makedirs(os.path.dirname(path_predictions), exist_ok=True)
                creat_pred_csv(path_predictions, best_test_preds)
                path_predictions_050 = "src/models/hiergat/output_en/prediction/" + str(run_tag) + '_predictions_050.csv'
                os.makedirs(os.path.dirname(path_predictions_050), exist_ok=True)
                creat_pred_csv(path_predictions_050, best_test_preds_050)
                path_predictions_100 = "src/models/hiergat/output_en/prediction/" + str(run_tag) + '_predictions_100.csv'
                os.makedirs(os.path.dirname(path_predictions_100), exist_ok=True)
                creat_pred_csv(path_predictions_100, best_test_preds_100)

            print("current_best_test_f1: " + str(best_test_f1) + ", current_best_test_f1_050: " + str(best_test_f1_050) + ", current_best_test_f1_100: " + str(best_test_f1_100))

        else:
            no_improvement_count += 1

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Amazon-Google")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="src/models/hiergat/checkpoints_en/")
    parser.add_argument("--lm_path", type=str, default=None)
    parser.add_argument("--split", dest="split", action="store_true")
    parser.add_argument("--lm", type=str, default='bert')

    args = parser.parse_args()

    # only a single task for baseline
    task = args.task

    # create the tag of the run
    run_tag = '%s_lr=%s_id=%d_batch=%d_lm=%s_english' % (task, args.lr, args.run_id, args.batch_size, args.lm)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('src/models/hiergat/task_en.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    category = config['category']
    testset050 = config['testset050']
    testset100 = config['testset100']

    # load train/dev/test sets
    train_dataset = Dataset(trainset, category, lm=args.lm, lm_path=args.lm_path, max_len=args.max_len, split=args.split)
    valid_dataset = Dataset(validset, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset = Dataset(testset, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset050 = Dataset(testset050, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset100 = Dataset(testset100, category, lm=args.lm, lm_path=args.lm_path, split=args.split)

    job_name = f"{args.task}_seed{args.run_id}_english"

    run_with_tracking(
        job_name=job_name,
        func=initialize_and_train,
        trainset=train_dataset,
        validset=valid_dataset,
        testset=test_dataset,
        testset050=test_dataset050,
        testset100=test_dataset100,
        attr_num=train_dataset.get_attr_num(),
        args=args,
        run_tag=run_tag
    )