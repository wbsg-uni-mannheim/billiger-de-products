"""
Run pair-wise fine-tuning
"""
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd
from sklearn.metrics import classification_report, f1_score

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import time
import math

from copy import deepcopy

import torch
from torch import nn

import transformers as transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from dataset_en import BaselineClassificationDataset
from metrics import compute_metrics_baseline

from transformers import EarlyStoppingCallback

from transformers.utils.hp_naming import TrialShortNamer

from pdb import set_trace


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

from codecarbon import OfflineEmissionsTracker
import time
import os
import json
import pandas as pd
from memory_profiler import memory_usage

def save_predictions_roberta_style(
    dataset,
    predictions,   # shape (N, 2)
    label_ids,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logits = np.asarray(predictions)

    # probability of positive class (class 1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    # SAME as compute_metrics_baseline
    preds = np.argmax(logits, axis=1)

    # pair_id extraction
    if "pair_id" in dataset.data.columns:
        pair_ids = dataset.data["pair_id"].astype(str).tolist()
    else:
        pair_ids = list(range(len(preds)))  # fallback

    if label_ids is None:
        labels = [None] * len(preds)
    else:
        labels = np.asarray(label_ids).astype(int)

    df = pd.DataFrame({
        "pair_id": pair_ids,
        "label": labels,
        "probability": probs,
        "prediction": preds,
    })

    df.to_csv(output_path, index=False)
    
# ========================
# Efficiency Tracker
# ========================
def run_with_tracking(job_name, func, *args, electricity_price_eur_per_kwh=0.30, **kwargs):

    os.makedirs("data/efficiency_tracker/roberta_en", exist_ok=True)
    csv_path = f"data/efficiency_tracker/roberta_en/{job_name}.csv"
    json_path = f"data/efficiency_tracker/roberta_en/{job_name}.json"

    tracker = OfflineEmissionsTracker(
        country_iso_code="DEU",
        output_file=csv_path,
    )
# ---- GPU MEMORY RESET (BEFORE TRAINING) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    tracker.start()

    output = func(*args, **kwargs)

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

    return output

#MODEL_PARAMS=['frozen', 'pool', 'use_colcls', 'sum_axial']

###############################################################################
# CALIBRATION METHODS (PAPER-STYLE, OFFLINE AFTER TRAINING)                  #
# - Temperature scaling: grid search minimizing ECE on validation            #
# - MC Dropout: tune dropout p to minimize ECE (only classifier head)        #
# - Deep Ensemble: average predictions over multiple trained runs            #
###############################################################################

from sklearn.metrics import f1_score

# ------------------------- ECE (as used in the paper) ------------------------
def compute_ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece


# ---------------------------------------------------------------------------
# TEMPERATURE SCALING (grid search on validation ECE, with F1 constraint)
# ---------------------------------------------------------------------------
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # initialize T close to 1
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, **inputs):
        logits = self.model(**inputs).logits
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        Fits the temperature parameter T using NLL on validation data.
        """

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits_tensor / self.temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self.temperature.detach().cpu().item()

def fit_temperature(model, trainer, val_dataset):
    model.eval()

    trainer.data_collator = DataCollatorWithPadding(
        tokenizer=trainer.tokenizer,
        padding="longest",
        max_length=256,
    )

    loader = trainer.get_eval_dataloader(val_dataset)

    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch).logits
            logits_list.append(out.cpu().numpy())
            labels_list.append(batch["labels"].cpu().numpy())

    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)

    # Wrap model in temperature scaling module
    temp_model = ModelWithTemperature(model)

    T = temp_model.set_temperature(logits, labels)
    print(f"[Temperature Scaling] Learned T = {T:.4f}")

    return T

def predict_with_temperature(model, trainer, dataset, T):
    model.eval()
    trainer.data_collator = DataCollatorWithPadding(
        tokenizer=trainer.tokenizer,
        padding="longest",
        max_length=256,
    )

    loader = trainer.get_eval_dataloader(dataset)

    all_probs = []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**batch).logits / T
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_logits)



# ---------------------------------------------------------------------------
# MC DROPOUT
# - Only dropout in classifier head (we only change p there)
# - Tune p ∈ {0.1, 0.2, ..., 0.5} by minimizing ECE on validation
# ---------------------------------------------------------------------------
def set_classifier_dropout(model, p):
    for name, module in model.named_modules():
        if "classifier" in name and isinstance(module, nn.Dropout):
            module.p = p


def mc_predict(model, trainer, dataset, passes=20):
    # train() to activate dropout, but with no_grad so no gradients
    model.train()
    trainer.data_collator = DataCollatorWithPadding(
        tokenizer=trainer.tokenizer,
        padding="longest",
        max_length=256,
    )
    loader = trainer.get_eval_dataloader(dataset)

    all_passes = []
    all_passes_logits = []

    with torch.no_grad():
        for _ in range(passes):
            probs = []
            logits_list = []
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch).logits
                prob = torch.softmax(out, dim=1)[:, 1]
                probs.append(prob.cpu().numpy())
                logits_list.append(out.cpu().numpy())
            all_passes.append(np.concatenate(probs))
            all_passes_logits.append(np.concatenate(logits_list))

    return np.mean(np.stack(all_passes), axis=0), np.mean(np.stack(all_passes_logits), axis=0)


def tune_mc_dropout(model, trainer, val_dataset):
    # baseline logits
    trainer.data_collator = DataCollatorWithPadding(
        tokenizer=trainer.tokenizer,
        padding="longest",
        max_length=256,
    )
    loader = trainer.get_eval_dataloader(val_dataset)

    logits = []
    labels = []

    with torch.no_grad():
        model.eval()
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch).logits
            logits.append(out.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)

    base_probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    base_f1 = f1_score(labels, (base_probs >= 0.5).astype(int))

    best_p, best_ece = 0.1, 999.0

    for p in np.arange(0.05, 1.0, 0.05):
        print(f"[MC Dropout] Testing p={p}")
        set_classifier_dropout(model, p)
        preds, _ = mc_predict(model, trainer, val_dataset, passes=20)

        f1 = f1_score(labels, (preds >= 0.5).astype(int))
        if f1 < base_f1 - 0.02:
            continue

        ece = compute_ece(preds, labels)
        if ece < best_ece:
            best_ece = ece
            best_p = p

    print(f"[MC Dropout] BEST p = {best_p}, ECE = {best_ece}")
    return best_p


def predict_mc_dropout(model, trainer, dataset, p, passes=20):
    set_classifier_dropout(model, p)
    return mc_predict(model, trainer, dataset, passes=passes)


# ---------------------------------------------------------------------------
# DEEP ENSEMBLE (PAPER-STYLE EXPERIMENT, BUT USING YOUR 3 RUNS)
# - We treat each trained run (different seed) as one ensemble member
# - Ensemble prediction = average of member probabilities
# ---------------------------------------------------------------------------
def run_deep_ensemble_from_dirs(run_dirs, trainer, dataset, device):
    ensemble_probs = []
    ensemble_logits = []

    for d in run_dirs:
        print(f"[Ensemble] Loading model from {d}")
        model = AutoModelForSequenceClassification.from_pretrained(d).to(device)
        model.eval()

        trainer.model = model
        trainer.data_collator = DataCollatorWithPadding(
            tokenizer=trainer.tokenizer,
            padding="longest",
            max_length=256,
        )
        
        out = trainer.predict(dataset)
        logits = out.predictions
        ensemble_logits.append(logits)

        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        ensemble_probs.append(probs)

    return np.mean(np.stack(ensemble_probs), axis=0), np.mean(np.stack(ensemble_logits), axis=0)


###############################################################################
# END CALIBRATION BLOCK
###############################################################################





@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_pretrained_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    do_param_opt: Optional[bool] = field(
        default=False, metadata={"help": "If aou want to do hyperparamter optimization"}
    )
    frozen: Optional[str] = field(
        default='frozen', metadata={"help": "If encoder params should be frozen, options: frozen, unfrozen"}
    )
    grad_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If aou want to use gradient checkpointing"}
    )
    tokenizer: Optional[str] = field(
        default='huawei-noah/TinyBERT_General_4L_312D',
        metadata={
            "help": "Tokenizer to use"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    train_size: Optional[str] = field(
        default=None, metadata={"help": "The size of the training set."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    augment: Optional[str] = field(
        default=None, metadata={"help": "The data augmentation to use."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    dataset_name: Optional[str] = field(
        default='lspc',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    additional_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to additional data to be used for training"
        },
    )
    only_additional: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If only additional data should be used without domain training data"
        },
    )
    only_name: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use only the name attribute"
        },
    )
    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training file.")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def main():

    def get_posneg(train_dataset):
        counts = train_dataset.data['label'].value_counts()
        ratio = counts[0]/counts[1]
        return math.ceil(ratio)

    def model_init(trial):
        # if trial is not None:
        #     init_args = {k:v for k, v in trial.items() if k in MODEL_PARAMS}
        # else:
        #     init_args = {}
        init_args = {}
        #pos_neg = get_posneg(train_dataset)
        if model_args.model_pretrained_checkpoint:
            my_model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, num_labels=2)
            if model_args.grad_checkpoint:
                if hasattr(my_model, "bert"):
                    my_model.bert.gradient_checkpointing_enable()
                elif hasattr(my_model, "roberta"):
                    my_model.roberta.gradient_checkpointing_enable()
                else:
                    my_model.gradient_checkpointing_enable()
            return my_model
        else:
            my_model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, num_labels=2)
            if model_args.grad_checkpoint:
                if hasattr(my_model, "bert"):
                    my_model.bert.gradient_checkpointing_enable()
                elif hasattr(my_model, "roberta"):
                    my_model.roberta.gradient_checkpointing_enable()
                else:
                    my_model.gradient_checkpointing_enable()
            return my_model

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_num_workers = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    
        #if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        #    raise ValueError(
        #        f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #        "Use --overwrite_output_dir to overcome."
        #    )
        #elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        #    logger.info(
        #        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
        #        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        #    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = data_files

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset = BaselineClassificationDataset(train_dataset, dataset_type='train', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, aug=data_args.augment, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if training_args.evaluation_strategy != 'no':
            validation_dataset = raw_datasets["validation"]
            validation_dataset = BaselineClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if training_args.load_best_model_at_end:
            test_dataset = raw_datasets["test"]
            test_dataset = BaselineClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
            if data_args.dataset_name == 'lspc' and 'products' in raw_datasets["train"]:
                unseen_set_one = BaselineClassificationDataset(raw_datasets["test"].replace('000un', '050un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
                unseen_set_two = BaselineClassificationDataset(raw_datasets["test"].replace('000un', '100un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        validation_dataset = raw_datasets["validation"]
        validation_dataset = BaselineClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)

    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        test_dataset = BaselineClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if data_args.dataset_name == 'lspc' and 'products' in raw_datasets["train"]:
                unseen_set_one = BaselineClassificationDataset(raw_datasets["test"].replace('000un', '050un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
                unseen_set_two = BaselineClassificationDataset(raw_datasets["test"].replace('000un', '100un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer, padding='longest', max_length=256)

    # Early stopping callback
    callback = EarlyStoppingCallback(early_stopping_patience=10)

    if training_args.do_train and model_args.do_param_opt:

        from ray import tune
        def my_hp_space(trial):
            return {
                "learning_rate": tune.loguniform(5e-5, 5e-3),
                "warmup_ratio": tune.choice([0.05, 0.075, 0.10]),
                "max_grad_norm": tune.choice([0.0, 1.0]),
                "weight_decay": tune.loguniform(0.001, 0.1),
                "seed": tune.randint(1, 50)
            }

        def my_objective(metrics):
            return metrics['eval_f1']
        

        trainer = CustomTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_baseline,
        callbacks=[callback]
        )
        trainer.args.save_total_limit = 1

        trainer.pos_neg = get_posneg(train_dataset)

        def hp_name(trial):
            namer = TrialShortNamer()
            namer.set_defaults('hp', {'learning_rate': 1e-4, 'warmup_ratio': 0.0, 'max_grad_norm': 1.0, 'weight_decay': 0.01, 'seed':1})
            return namer.shortname(trial)

        # asha_scheduler = tune.schedulers.ASHAScheduler(
        #     time_attr='epoch',
        #     metric='eval_f1',
        #     mode='max',
        #     max_t=trainer.args.num_train_epochs,
        #     grace_period=15
        #     )
        initial_configs = [
            {
                "learning_rate": 1e-3,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            },
            {
                "learning_rate": 1e-4,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            }
            ]
                
        from ray.tune.suggest.hebo import HEBOSearch
        hebo = HEBOSearch(metric="eval_f1", mode="max", points_to_evaluate=initial_configs, random_state_seed=42)

        best_run = trainer.hyperparameter_search(n_trials=24, direction="maximize", hp_space=my_hp_space, compute_objective=my_objective, backend='ray', 
        resources_per_trial={'cpu':4,'gpu':1}, local_dir=f'{training_args.output_dir}ray_results/', hp_name=hp_name, search_alg=hebo)
        
        with open(f'{training_args.output_dir}best_params.json', 'w') as f:
            json.dump(best_run, f)

    output_dir = deepcopy(training_args.output_dir)
    for run in range(3):
        init_args = {}

        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f'{output_dir}{run}'
        # if model_args.do_param_opt:
        #     init_args = {k:v for k, v in best_run.hyperparameters.items() if k in MODEL_PARAMS}


        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("Num workers being used:", training_args.dataloader_num_workers)

        pos_neg = get_posneg(train_dataset)
        if model_args.model_pretrained_checkpoint:
            model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, num_labels=2)
            if model_args.grad_checkpoint:
                if hasattr(model, "bert"):
                    model.bert.gradient_checkpointing_enable()
                elif hasattr(model, "roberta"):
                    model.roberta.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_enable()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, num_labels=2)
            if model_args.grad_checkpoint:
                if hasattr(model, "bert"):
                    model.bert.gradient_checkpointing_enable()
                elif hasattr(model, "roberta"):
                    model.roberta.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_enable()

        model.resize_token_embeddings(len(train_dataset.tokenizer))

        # Initialize our Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics_baseline,
            callbacks=[callback]
        )
        trainer.pos_neg = get_posneg(train_dataset)
        # Training
        if training_args.do_train:
            if model_args.do_param_opt:
                for n, v in best_run.hyperparameters.items():
                    setattr(trainer.args, n, v)
                    # if n not in MODEL_PARAMS:
                    #     setattr(trainer.args, n, v)

            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            time.sleep(30)
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics(f"train", metrics)
            trainer.save_metrics(f"train", metrics)
            trainer.save_state()
        
        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                metric_key_prefix="eval"
            )
            max_eval_samples = len(validation_dataset)
            metrics["eval_samples"] = max_eval_samples

            trainer.log_metrics(f"eval", metrics)
            trainer.save_metrics(f"eval", metrics)

        if training_args.do_predict or training_args.do_train:
            logger.info("*** Predict ***")

            predict_results = run_with_tracking(
                job_name=f"roberta_predict_000un_{data_args.train_size}_{data_args.dataset_name}_id={run}",
                func=trainer.predict,
                test_dataset=test_dataset, metric_key_prefix="predict"
            )

            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = max_predict_samples

            trainer.log_metrics(f"predict", metrics)
            trainer.save_metrics(f"predict", metrics)

            if 'products' in raw_datasets["train"]:
                predict_results = run_with_tracking(
                    job_name=f"roberta_predict_050un_{data_args.train_size}_{data_args.dataset_name}_id={run}",
                    func=trainer.predict,
                    test_dataset=unseen_set_one,
                    metric_key_prefix="predict_un050",
                )

                metrics = predict_results.metrics
                max_predict_samples = len(unseen_set_one)
                metrics["predict_samples_un050"] = max_predict_samples

                trainer.log_metrics(f"predict_un050", metrics)
                trainer.save_metrics(f"predict_un050", metrics)

                predict_results = run_with_tracking(
                    job_name=f"roberta_predict_100un_{data_args.train_size}_{data_args.dataset_name}_id={run}",
                    func=trainer.predict,
                    test_dataset=unseen_set_two,
                    metric_key_prefix="predict_un100",
                )

                metrics = predict_results.metrics
                max_predict_samples = len(unseen_set_two)
                metrics["predict_samples_un100"] = max_predict_samples

                trainer.log_metrics(f"predict_un100", metrics)
                trainer.save_metrics(f"predict_un100", metrics)

    # ========================================================================
    # POST-TRAINING CONFIDENCE CALIBRATION (PAPER-STYLE, OFFLINE)
    # ========================================================================
    if training_args.do_train and training_args.load_best_model_at_end:
        print(">>> Running post-training calibration for all runs")

        device = training_args.device
        run_dirs = [f"{output_dir}{i}" for i in range(3)]

        # FIX: ensure trainer has tokenizer + collator for calibration
        trainer.tokenizer = train_dataset.tokenizer
        trainer.data_collator = DataCollatorWithPadding(
            tokenizer=train_dataset.tokenizer,
            padding="longest",
            max_length=256,
        )
        trainer.pos_neg = get_posneg(train_dataset)

        for d in run_dirs:
            print(f"\n[Calibration] Processing run directory: {d}")
            model = AutoModelForSequenceClassification.from_pretrained(d).to(device)

            # --------- 1) Save BASE probabilities on test set ----------
            trainer.model = model
            base_out = trainer.predict(test_dataset)
            base_logits = base_out.predictions
            base_probs = torch.softmax(torch.tensor(base_logits), dim=1)[:, 1].numpy()
            np.save(f"{d}/base_probs.npy", base_probs)
            np.save(f"{d}/base_logits.npy", base_logits)


            base_out_50 = trainer.predict(unseen_set_one)
            base_logits_50 = base_out_50.predictions
            base_probs_50 = torch.softmax(torch.tensor(base_logits_50), dim=1)[:, 1].numpy()
            np.save(f"{d}/base_probs_un050.npy", base_probs_50)
            np.save(f"{d}/base_logits_un050.npy", base_logits_50)

            base_out_100 = trainer.predict(unseen_set_two)
            base_logits_100 = base_out_100.predictions
            base_probs_100 = torch.softmax(torch.tensor(base_logits_100), dim=1)[:, 1].numpy()
            np.save(f"{d}/base_probs_un100.npy", base_probs_100)
            np.save(f"{d}/base_logits_un100.npy", base_logits_100)

            # --------- 2) Temperature scaling (fit on validation) ------
            T = fit_temperature(model, trainer, validation_dataset)
            temp_probs, temp_logits = predict_with_temperature(model, trainer, test_dataset, T)
            np.save(f"{d}/temperature_scaled_probs.npy", temp_probs)
            np.save(f"{d}/temperature_scaled_logits.npy", temp_logits)

            temp_probs_50, temp_logits_50 = predict_with_temperature(model, trainer, unseen_set_one, T)
            np.save(f"{d}/temperature_scaled_probs_un050.npy", temp_probs_50)
            np.save(f"{d}/temperature_scaled_logits_un050.npy", temp_logits_50)

            temp_probs_100, temp_logits_100 = predict_with_temperature(model, trainer, unseen_set_two, T)
            np.save(f"{d}/temperature_scaled_probs_un100.npy", temp_probs_100)
            np.save(f"{d}/temperature_scaled_logits_un100.npy", temp_logits_100)


            # --------- 3) MC Dropout (tune p on validation) ------------
            best_p = tune_mc_dropout(model, trainer, validation_dataset)
            mc_probs, mc_predictlogits = predict_mc_dropout(model, trainer, test_dataset, best_p, passes=20)
            np.save(f"{d}/mc_dropout_probs.npy", mc_probs)
            np.save(f"{d}/mc_dropout_logits.npy", mc_predictlogits)

            mc_probs_50, mc_predictlogits_50 = predict_mc_dropout(model, trainer, unseen_set_one, best_p)
            np.save(f"{d}/mc_dropout_probs_un050.npy", mc_probs_50)
            np.save(f"{d}/mc_dropout_logits_un050.npy", mc_predictlogits_50)

            mc_probs_100, mc_predictlogits_100 = predict_mc_dropout(model, trainer, unseen_set_two, best_p)
            np.save(f"{d}/mc_dropout_probs_un100.npy", mc_probs_100)
            np.save(f"{d}/mc_dropout_logits_un100.npy", mc_predictlogits_100)


        # ----------------- 4) Deep ensemble over the 3 runs -------------
        print("\n>>> Building deep ensemble over all runs")
        # model in trainer will be replaced inside this function
        ens_probs, ens_logits = run_deep_ensemble_from_dirs(run_dirs, trainer, test_dataset, device)
        np.save(f"{output_dir}/ensemble_probs.npy", ens_probs)
        np.save(f"{output_dir}/ensemble_logits.npy", ens_logits)

        ens_probs_50, ens_logits_50  = run_deep_ensemble_from_dirs(run_dirs, trainer, unseen_set_one, device)
        np.save(f"{output_dir}/ensemble_probs_un050.npy", ens_probs_50)
        np.save(f"{output_dir}/ensemble_logits_un050.npy", ens_logits_50)

        ens_probs_100, ens_logits_100 = run_deep_ensemble_from_dirs(run_dirs, trainer, unseen_set_two, device)
        np.save(f"{output_dir}/ensemble_probs_un100.npy", ens_probs_100)
        np.save(f"{output_dir}/ensemble_logits_un100.npy", ens_logits_100)

    # ========================================================================

    return results

if __name__ == "__main__":
    main()
