import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from .dataset import DittoDataset
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..', '..')))
from src.cross_language.predictions import write_per_pair_predictions
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
# from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='bert', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)
        print("[INIT DEBUG] fc weight mean=", self.fc.weight.mean().item())



    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            if random.random() < 0.01:
                print("[LAM DEBUG] lambda=", float(aug_lam))

            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()

# TODO changed by Kseni added , return_preds=False
def evaluate(model, iterator, threshold=None, return_preds=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()
        print("[PRED DEBUG] mean prob=", np.mean(all_probs),
        "min=", np.min(all_probs),
        "max=", np.max(all_probs),
        "pos@0.5=", sum(p>0.5 for p in all_probs), "/", len(all_probs))


    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        preds = {
            "labels": all_y,
            "probs": all_probs,
            "preds": pred,
            "threshold": threshold
        }
        if return_preds:
            return f1, preds
        else:
            return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1: #type: ignore
                f1 = new_f1
                best_th = th
        
        if return_preds:
            return f1, best_th, {
            "labels": all_y,
            "probs": all_probs,
            "preds": [1 if p > best_th else 0 for p in all_probs],
            "threshold": best_th
            }
        else:
            return f1, best_th
        
#TODO added by Ksenia to save predictions
def creat_pred_csv(path_predictions, best_test_preds):
    os.makedirs(os.path.dirname(path_predictions), exist_ok=True)
    with open(path_predictions, "w") as f:
        f.write("label,probability,prediction,threshold\n")
        for i in range(len(best_test_preds['labels'])):
            f.write(f"{best_test_preds['labels'][i]},{best_test_preds['probs'][i]},{best_test_preds['preds'][i]},{best_test_preds['threshold']}\n")

def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)

        loss = criterion(prediction, y.to(model.device))
        if i < 30:  # nur die ersten paar Batches
            print(f"[BATCH DEBUG {i}] Positives:", int(y.sum()), "/", len(y))
        if hp.fp16:
            pass
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(
    trainset,
    validset,
    testset,
    testset050,
    testset100,
    run_tag,
    hp,
    path,
    extra_testsets=None,
    extra_pair_sources=None,
):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        testset050 (DittoDataset): the 50% test set
        testset100 (DittoDataset): the 100% test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    """run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d_adjusted' % (hp.task, hp.lm, hp.da,
                hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')"""
    output_file = os.path.join(path, f"{run_tag}.txt")
    os.makedirs(path, exist_ok=True)

    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter050 = data.DataLoader(dataset=testset050,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter100 = data.DataLoader(dataset=testset100,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    extra_test_iters = {
        name: data.DataLoader(
            dataset=dataset,
            batch_size=hp.batch_size * 16,
            shuffle=False,
            num_workers=0,
            collate_fn=padder,
        )
        for name, dataset in (extra_testsets or {}).items()
    }
    
    ys = [y for _,y in valid_iter.dataset]
    print("[DEV DEBUG] positives:", sum(ys), "/", len(ys))

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        pass
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = best_test_f1_050 = best_test_f1_100 = 0.0
    best_extra_metrics = {
        name: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for name in extra_test_iters
    }
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter) #type: ignore
        test_f1, test_preds = evaluate(model, test_iter, threshold=th, return_preds=True) #type: ignore
        test_f1_050, test_preds_050 = evaluate(model, test_iter050, threshold=th, return_preds=True) #type: ignore
        test_f1_100, test_preds_100 = evaluate(model, test_iter100, threshold=th, return_preds=True) #type: ignore

        if dev_f1 > best_dev_f1:#type: ignore
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_test_preds = test_preds
            best_test_f1_050 = test_f1_050
            best_test_f1_100 = test_f1_100
            best_test_preds_050 = test_preds_050
            best_test_preds_100 = test_preds_100
            for name, iterator in extra_test_iters.items():
                f1, predictions = evaluate(
                    model,
                    iterator,
                    threshold=th,
                    return_preds=True,
                )
                best_extra_metrics[name] = {
                    "precision": metrics.precision_score(
                        predictions["labels"],
                        predictions["preds"],
                        zero_division=0,
                    ),
                    "recall": metrics.recall_score(
                        predictions["labels"],
                        predictions["preds"],
                        zero_division=0,
                    ),
                    "f1": f1,
                }
                # Aggregate F1 alone cannot be rescored if the labels turn out to
                # be wrong, so persist the per-pair layer keyed by pair_id.
                source = (extra_pair_sources or {}).get(name)
                if source:
                    write_per_pair_predictions(
                        _os.path.join(hp.output_dir, f"{run_tag}_cross_{name}_predictions.csv"),
                        source,
                        predictions["labels"],
                        predictions["probs"],
                        predictions["preds"],
                    )
            
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    result = {
        'best_f1': best_test_f1,
        'best_f1_050': best_test_f1_050,
        'best_f1_100': best_test_f1_100,
    }
    for name, values in best_extra_metrics.items():
        for metric, value in values.items():
            result[f"best_{metric}_cross_{name}"] = value
    with open(output_file, "w") as f:
        f.write(repr(result) + '\n')

    writer.close()
