"""
Run contrastive pair-wise fine-tuning (TRAIN + PREDICT SEPARATELY)
"""

# ========================
# Imports
# ========================
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd
from sklearn.metrics import classification_report

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

import transformers as transformers

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

from modeling import ContrastiveClassifierModel
from dataset import ContrastiveClassificationDataset
from data_collators import DataCollatorContrastiveClassification
from metrics import compute_metrics_bce

from transformers import EarlyStoppingCallback

from transformers.utils.hp_naming import TrialShortNamer

from pdb import set_trace

from codecarbon import OfflineEmissionsTracker

# ========================
# Reproducibility
# ========================
np.random.seed(42)
random.seed(42)

logger = logging.getLogger(__name__)

# ========================
# Arguments
# ========================
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

def save_predictions_ditto_style(
    dataset,
    predictions,
    label_ids,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # logits → probabilities
    logits = predictions.squeeze()

    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # MATCHES compute_metrics_bce
    preds = (logits >= 0.5).astype(int)

    # pair_id extraction (robust)
    if "pair_id" in dataset.data.columns:
        pair_ids = dataset.data["pair_id"].astype(str).tolist()
    else:
        pair_ids = list(range(len(probs)))  # fallback, should not happen

    if label_ids is None:
        labels = [None] * len(probs)
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

    os.makedirs("data/efficiency_tracker/r_supCon", exist_ok=True)
    csv_path = f"data/efficiency_tracker/r_supCon/{job_name}.csv"
    json_path = f"data/efficiency_tracker/r_supCon/{job_name}.json"

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


# ========================
# Helpers
# ========================
def get_posneg(dataset):
    counts = dataset.data["labels"].value_counts()
    return math.ceil(counts[0] / counts[1])

# ========================
# TRAIN
# ========================
def train_model(model_args, data_args, training_args, run, output_dir):

    def model_init(trial):
        # if trial is not None:
        #     init_args = {k:v for k, v in trial.items() if k in MODEL_PARAMS}
        # else:
        #     init_args = {}
        init_args = {}
        pos_neg = get_posneg(train_dataset)
        if model_args.model_pretrained_checkpoint:
            my_model = ContrastiveClassifierModel(checkpoint_path=model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                my_model.encoder.transformer._set_gradient_checkpointing(my_model.encoder.transformer.encoder, True)
            return my_model
        else:
            my_model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                my_model.encoder.transformer._set_gradient_checkpointing(my_model.encoder.transformer.encoder, True)
            return my_model
    set_seed(training_args.seed)
    init_args = {}
    train_dataset = ContrastiveClassificationDataset(
        data_args.train_file,
        dataset_type="train",
        size=data_args.train_size,
        tokenizer=model_args.tokenizer,
        dataset=data_args.dataset_name,
        aug=data_args.augment,
        additional_data=data_args.additional_data,
        only_additional=data_args.only_additional,
        only_name=data_args.only_name,
    )

    val_dataset = ContrastiveClassificationDataset(
        data_args.validation_file,
        dataset_type="validation",
        size=data_args.train_size,
        tokenizer=model_args.tokenizer,
        dataset=data_args.dataset_name,
        additional_data=data_args.additional_data,
        only_additional=data_args.only_additional,
        only_name=data_args.only_name,
    )

    pos_neg = get_posneg(train_dataset)
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = data_files
    # Data collator
    data_collator = DataCollatorContrastiveClassification(tokenizer=train_dataset.tokenizer)

    # Early stopping callback
    callback = EarlyStoppingCallback(early_stopping_patience=10)
    best_run = None
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

        trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_bce,
        callbacks=[callback]
        )
        trainer.args.save_total_limit = 1

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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    pos_neg = get_posneg(train_dataset)
    if model_args.model_pretrained_checkpoint:
        model = ContrastiveClassifierModel(checkpoint_path=model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
        if model_args.grad_checkpoint:
            model.encoder.transformer.gradient_checkpointing_enable()
    else:
        model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
        if model_args.grad_checkpoint:
            model.encoder.transformer.gradient_checkpointing_enable()


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorContrastiveClassification(
            tokenizer=train_dataset.tokenizer
        ),
        compute_metrics=compute_metrics_bce,
        callbacks=[EarlyStoppingCallback(10)],
    )

    # Training
    if training_args.do_train:
        if model_args.do_param_opt and best_run is not None:
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

# ========================
# Evaluate (ONE DATASET)
# ========================
def eval_dataset(dataset_path, dataset_label, model_args, data_args, output_dir):

    
    validation_dataset = ContrastiveClassificationDataset(
        dataset_path,
        dataset_type='validation',
        size=data_args.train_size,
        tokenizer=model_args.tokenizer,
        dataset=data_args.dataset_name,
        additional_data=data_args.additional_data,
        only_additional=data_args.only_additional,
        only_name=data_args.only_name,
    )
    pos_neg = get_posneg(validation_dataset)

    model = ContrastiveClassifierModel.from_pretrained(
        output_dir,
        len_tokenizer=len(validation_dataset.tokenizer),
        checkpoint_path=model_args.model_pretrained_checkpoint,
        model=model_args.tokenizer,
        frozen=model_args.frozen,
        pos_neg=pos_neg
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=32,
            do_train=False,
            do_predict=True,
        ),
        eval_dataset=validation_dataset,  # <-- add this
        data_collator=DataCollatorContrastiveClassification(
            tokenizer=validation_dataset.tokenizer
        ),
        compute_metrics=compute_metrics_bce,
    )


    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(
        metric_key_prefix="eval"
    )
    max_eval_samples = len(validation_dataset)
    metrics["eval_samples"] = max_eval_samples

    trainer.log_metrics(f"eval", metrics)
    trainer.save_metrics(f"eval", metrics)

# ========================
# PREDICT (ONE DATASET)
# ========================
def predict_dataset(dataset_path, dataset_label, model_args, data_args, output_dir, unseen, run):
    
    dataset = ContrastiveClassificationDataset(
            dataset_path,
            dataset_type="test",
            size=data_args.train_size,
            tokenizer=model_args.tokenizer,
            dataset=data_args.dataset_name,
            additional_data=data_args.additional_data,
            only_additional=data_args.only_additional,
            only_name=data_args.only_name,
        )

    pos_neg = get_posneg(dataset)

    model = ContrastiveClassifierModel.from_pretrained(
        output_dir,
        len_tokenizer=len(dataset.tokenizer),
        checkpoint_path=model_args.model_pretrained_checkpoint,
        model=model_args.tokenizer,
        frozen=model_args.frozen,
        pos_neg=pos_neg
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=32,
            do_train=False,
            do_predict=True,
        ),
        data_collator=DataCollatorContrastiveClassification(
            tokenizer=dataset.tokenizer
        ),
        compute_metrics=compute_metrics_bce,
    )

    predict_results = trainer.predict(dataset, metric_key_prefix=f"predict_{unseen}")
    metrics = predict_results.metrics
    max_predict_samples = len(dataset)
    metrics[f"predict_samples_{unseen}"] = max_predict_samples

    trainer.log_metrics(f"predict_{unseen}", metrics)
    trainer.save_metrics(f"predict_{unseen}", metrics)

    # =========================
    # SAVE PREDICTIONS (HERE!)
    # =========================
    #data_args.testfile is path, so i need the last part without extension for the name
    test_name =data_args.test_file.split("/")[-1].split(".")[0]
    pred_out = os.path.join(
        "src/models/r-supCon/reports/predictions",
        f"{dataset_label}_{unseen}_{run}_{test_name}_{data_args.train_size}.csv"
    )

    save_predictions_ditto_style(
        dataset=dataset,
        predictions=predict_results.predictions,
        label_ids=predict_results.label_ids,
        output_path=pred_out,
    )

    print(f"Saved predictions to: {pred_out}")


# ========================
# MAIN
# ========================
def main():
    
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.frozen = False
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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    output_dir = deepcopy(training_args.output_dir)

    for run in range(3):
        
        init_args = {}

        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f'{output_dir}{run}'

        test_name =data_args.test_file.split("/")[-1].split(".")[0]

        # --------------------
        # TRAIN (tracked once)
        # --------------------
        run_with_tracking(
            job_name=f"train_{data_args.train_size}_{test_name}_id={run}_lm=roberta",
            func=train_model,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            run=run,
            output_dir=training_args.output_dir
        )


        # --------------------
        # EVALUATE (tracked separately)
        # --------------------
        if training_args.do_eval:

            run_with_tracking(
                job_name=f"eval_{data_args.train_size}_{test_name}_id={run}_lm=roberta",
                func=eval_dataset,
                dataset_path=data_args.validation_file,
                dataset_label="validation",
                model_args=model_args,
                data_args=data_args,
                output_dir=training_args.output_dir
            )
        # --------------------
        # PREDICT (tracked separately)
        # --------------------
        
        if training_args.do_predict or training_args.do_train:
            run_with_tracking(
                job_name=f"predict_un000_{data_args.train_size}_{test_name}_id={run}_lm=roberta",
                func=predict_dataset,
                dataset_path=data_args.test_file,
                dataset_label="test",
                model_args=model_args,
                data_args=data_args,
                output_dir=training_args.output_dir,
                unseen="un000",
                run = run
            )
            
            run_with_tracking(
                job_name=f"predict_un050_{data_args.train_size}_{test_name}_id={run}_lm=roberta",
                func=predict_dataset,
                dataset_path=data_args.test_file.replace("000un", "050un"),
                dataset_label="un050",
                model_args=model_args,
                data_args=data_args,
                output_dir=training_args.output_dir,
                unseen="un050",
                run = run
            )

            run_with_tracking(
                job_name=f"predict_un100_{data_args.train_size}_{test_name}_id={run}_lm=roberta",
                func=predict_dataset,
                dataset_path=data_args.test_file.replace("000un", "100un"),
                dataset_label="un100",
                model_args=model_args,
                data_args=data_args,
                output_dir=training_args.output_dir,
                unseen="un100",
                run = run
            )


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    main()
