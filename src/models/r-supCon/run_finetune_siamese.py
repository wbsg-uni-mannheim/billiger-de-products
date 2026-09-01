"""
Run contrastive pair-wise fine-tuning
"""
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
from pathlib import Path
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
from src.cross_language.predictions import write_per_pair_predictions
from typing import Optional
import json

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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

#MODEL_PARAMS=['frozen', 'pool', 'use_colcls', 'sum_axial']

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_pretrained_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    cross_language_test_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing aligned cross-language pickle test sets."},
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



def main():

    def get_posneg(train_dataset):
        counts = train_dataset.data['labels'].value_counts()
        ratio = counts[0]/counts[1]
        return math.ceil(ratio)

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

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    """if model_args.frozen == 'frozen':
        model_args.frozen = True
    else:
        model_args.frozen = False"""
    model_args.frozen = False
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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
    cross_language_datasets = {}
    cross_language_pair_sources = {}

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset = ContrastiveClassificationDataset(train_dataset, dataset_type='train', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, aug=data_args.augment, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if training_args.evaluation_strategy != 'no':
            validation_dataset = raw_datasets["validation"]
            validation_dataset = ContrastiveClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if training_args.load_best_model_at_end:
            test_dataset = raw_datasets["test"]
            test_dataset = ContrastiveClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
            if data_args.dataset_name == 'lspc' and 'products' in raw_datasets["train"]:
                unseen_set_one = ContrastiveClassificationDataset(raw_datasets["test"].replace('000un', '050un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
                unseen_set_two = ContrastiveClassificationDataset(raw_datasets["test"].replace('000un', '100un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        validation_dataset = raw_datasets["validation"]
        validation_dataset = ContrastiveClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)

    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        test_dataset = ContrastiveClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
        if data_args.dataset_name == 'lspc' and 'products' in raw_datasets["train"]:
                unseen_set_one = ContrastiveClassificationDataset(raw_datasets["test"].replace('000un', '050un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)
                unseen_set_two = ContrastiveClassificationDataset(raw_datasets["test"].replace('000un', '100un'), dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_name=data_args.only_name)

    if data_args.cross_language_test_dir:
        for path in sorted(Path(data_args.cross_language_test_dir).glob("*.pkl.gz")):
            variant = next(
                name
                for name in ("de_de", "de_en", "en_de", "en_en", "random")
                if f"_{name}.pkl.gz" in path.name
            )
            cross_language_pair_sources[variant] = str(path)
            cross_language_datasets[variant] = ContrastiveClassificationDataset(
                str(path),
                dataset_type="test",
                size=data_args.train_size,
                tokenizer=model_args.tokenizer,
                dataset=data_args.dataset_name,
                additional_data=data_args.additional_data,
                only_additional=data_args.only_additional,
                only_name=data_args.only_name,
            )
    
    # Data collator
    data_collator = DataCollatorContrastiveClassification(tokenizer=train_dataset.tokenizer)

    # Early stopping callback
    callback = EarlyStoppingCallback(early_stopping_patience=10)

    output_dir = deepcopy(training_args.output_dir)
    for run in range(3):
        init_args = {}

        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f'{output_dir}{run}'

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

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics_bce,
            callbacks=[callback]
        )
        
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
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

            predict_results = trainer.predict(
                test_dataset,
                metric_key_prefix="predict"
            )

            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = max_predict_samples

            trainer.log_metrics(f"predict", metrics)
            trainer.save_metrics(f"predict", metrics)

            if 'products' in raw_datasets["train"]:
                predict_results = trainer.predict(
                    unseen_set_one,
                    metric_key_prefix="predict_un050"
                )

                metrics = predict_results.metrics
                max_predict_samples = len(unseen_set_one)
                metrics["predict_samples_un050"] = max_predict_samples

                trainer.log_metrics(f"predict_un050", metrics)
                trainer.save_metrics(f"predict_un050", metrics)

                predict_results = trainer.predict(
                    unseen_set_two,
                    metric_key_prefix="predict_un100"
                )

                metrics = predict_results.metrics
                max_predict_samples = len(unseen_set_two)
                metrics["predict_samples_un100"] = max_predict_samples

                trainer.log_metrics(f"predict_un100", metrics)
                trainer.save_metrics(f"predict_un100", metrics)

            for variant, dataset in cross_language_datasets.items():
                predict_results = trainer.predict(
                    dataset,
                    metric_key_prefix=f"predict_cross_{variant}",
                )
                metrics = predict_results.metrics
                metrics[f"predict_cross_{variant}_samples"] = len(dataset)
                trainer.log_metrics(f"predict_cross_{variant}", metrics)
                trainer.save_metrics(f"predict_cross_{variant}", metrics)
                # Per-pair layer so the cell can be rescored without retraining.
                import numpy as _np
                # R-SupCon's head emits one sigmoid score per pair and
                # compute_metrics_bce thresholds it at 0.5; mirror that exactly
                # so the dumped predictions reproduce the reported metrics.
                _scores = _np.asarray(predict_results.predictions).reshape(-1)
                _preds = (_scores >= 0.5).astype(int)
                write_per_pair_predictions(
                    _os.path.join(training_args.output_dir, f"predictions_cross_{variant}.csv"),
                    cross_language_pair_sources[variant],
                    _np.asarray(predict_results.label_ids).reshape(-1).astype(int),
                    _scores,
                    _preds,
                )
    return results

if __name__ == "__main__":
    main()
