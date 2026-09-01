import argparse
import json
import os
from pathlib import Path

from model.eval import eval_classifier, eval_on_task
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
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
from src.cross_language.predictions import write_per_pair_predictions

import sys
import random

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


def initialize_and_train(
    trainset,
    validset,
    testset,
    testset050,
    testset100,
    attr_num,
    args,
    run_tag,
    extra_testsets=None,
    extra_pair_sources=None,
):
    padder = Dataset.pad
    valid_iter = data.DataLoader(dataset=validset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, collate_fn=padder)
    test_iter050 = data.DataLoader(dataset=testset050, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=padder)
    test_iter100 = data.DataLoader(dataset=testset100, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=padder)
    extra_test_iters = {
        name: data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=padder,
        )
        for name, dataset in (extra_testsets or {}).items()
    }

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
    best_extra_metrics = {
        name: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for name in extra_test_iters
    }
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
                best_test_preds = test_preds
                best_test_f1_050 = test_f1_050
                best_test_preds_050 = test_preds_050
                best_test_f1_100 = test_f1_100
                best_test_preds_100 = test_preds_100
                for name, iterator in extra_test_iters.items():
                    _, precision, recall, f1, _, preds = eval_classifier(
                        model, iterator, return_preds=True
                    )
                    best_extra_metrics[name] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                    # Persist the per-pair layer so the cell can be rescored
                    # against corrected labels without retraining.
                    source = (extra_pair_sources or {}).get(name)
                    if source:
                        write_per_pair_predictions(
                            _os.path.join(args.output_dir, f"{run_tag}_cross_{name}_predictions.csv"),
                            source,
                            preds["labels"],
                            None,
                            preds["preds"],
                        )

            print("current_best_test_f1: " + str(best_test_f1) + ", current_best_test_f1_050: " + str(best_test_f1_050) + ", current_best_test_f1_100: " + str(best_test_f1_100))

            if args.save_model:
                pass
#             if args.save_model:
#                 if dev_f1 > best_dev_f1:
#                     best_dev_f1 = dev_f1
#                     torch.save(model.state_dict(), run_tag + '_dev.pt')
#                 if test_f1 > best_test_f1:
#                     best_test_f1 = dev_f1
#                     torch.save(model.state_dict(), run_tag + '_test.pt')
        else:
            no_improvement_count += 1

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'best_test_f1': best_test_f1,
        'best_test_f1_050': best_test_f1_050,
        'best_test_f1_100': best_test_f1_100,
    }
    for name, values in best_extra_metrics.items():
        for metric, value in values.items():
            result[f"best_test_{metric}_cross_{name}"] = value
    with open(os.path.join(args.output_dir, f'{run_tag}.txt'), "w") as f:
        f.write(repr(result) + '\n')
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
    parser.add_argument("--logdir", type=str, default="src/models/hiergat/checkpoints/")
    parser.add_argument("--lm_path", type=str, default=None)
    parser.add_argument("--output_dir", default="results/generated/hiergat/de")
    parser.add_argument("--split", dest="split", action="store_true")
    # roberta-base is the backbone for every reported Ditto/HierGAT number in
    # both the main grid (all_runs*.py) and the cross-language runs; the old
    # 'bert' default was never used and only made a silent mismatch possible.
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--cross_language_test_dir", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)

    args = parser.parse_args()

    # only a single task for baseline
    task = args.task

    # create the tag of the run
    run_tag = '%s_lr=%s_id=%d_batch=%d_lm=%s_adjusted' % (task, args.lr, args.run_id, args.batch_size, args.lm)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('src/models/hiergat/task.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = args.validation_file or config['validset']
    testset = config['testset']
    category = config['category']
    testset050 = config['testset050']
    testset100 = config['testset100']
    if args.cross_language_test_dir:
        testset = testset050
        testset100 = testset050

    # load train/dev/test sets
    train_dataset = Dataset(trainset, category, lm=args.lm, lm_path=args.lm_path, max_len=args.max_len, split=args.split)
    valid_dataset = Dataset(validset, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset = Dataset(testset, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset050 = Dataset(testset050, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    test_dataset100 = Dataset(testset100, category, lm=args.lm, lm_path=args.lm_path, split=args.split)
    cross_language_datasets = {}
    cross_language_pair_sources = {}
    if args.cross_language_test_dir:
        for path in sorted(Path(args.cross_language_test_dir).glob("*_gs_*.txt")):
            variant = next(
                name
                for name in ("de_de", "de_en", "en_de", "en_en", "random")
                if f"_{name}.txt" in path.name
            )
            cross_language_datasets[variant] = Dataset(
                str(path),
                category,
                lm=args.lm,
                lm_path=args.lm_path,
                split=args.split,
            )
            # The serialized text has no pair_id; recover it from the pair file
            # that produced it, in the same row order.
            source = Path("data/processed_cross_language/gold-standards_adjusted") / (
                f"preprocessed_{path.stem}.pkl.gz"
            )
            if not source.exists():
                raise FileNotFoundError(
                    f"No pair file with pair_id for cross-language variant {variant}: {source}"
                )
            cross_language_pair_sources[variant] = str(source)

    initialize_and_train(
        trainset=train_dataset,
        validset=valid_dataset,
        testset=test_dataset,
        testset050=test_dataset050,
        testset100=test_dataset100,
        attr_num=train_dataset.get_attr_num(),
        args=args,
        run_tag=run_tag,
        extra_testsets=cross_language_datasets,
        extra_pair_sources=cross_language_pair_sources,
    )
