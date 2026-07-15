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
                best_test_preds = test_preds
                best_test_f1_050 = test_f1_050
                best_test_preds_050 = test_preds_050
                best_test_f1_100 = test_f1_100
                best_test_preds_100 = test_preds_100
            if (epoch == args.n_epochs):
                os.makedirs(args.output_dir, exist_ok=True)
                result = {
                    'best_test_f1': best_test_f1,
                    'best_test_f1_050': best_test_f1_050,
                    'best_test_f1_100': best_test_f1_100,
                }
                with open(os.path.join(args.output_dir, f'{run_tag}.txt'), "w") as f:
                    f.write(repr(result) + '\n')

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
    parser.add_argument("--lm", type=str, default='bert')

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

    initialize_and_train(
        trainset=train_dataset,
        validset=valid_dataset,
        testset=test_dataset,
        testset050=test_dataset050,
        testset100=test_dataset100,
        attr_num=train_dataset.get_attr_num(),
        args=args,
        run_tag=run_tag,
    )
