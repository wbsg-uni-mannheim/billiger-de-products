"""Evaluate an existing selected checkpoint on explicit v1.1 inputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('roberta', 'xlmr', 'r-supcon'), required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--inputs', nargs='+', required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
assert torch.cuda.is_available(), 'GPU allocation required'
args.output.mkdir(parents=True, exist_ok=True)
backbone = 'xlm-roberta-base' if args.model == 'xlmr' else 'roberta-base'
module = 'r-supCon' if args.model == 'r-supcon' else 'transformer_bert_confidence'
sys.path.insert(0, str(Path('src/models') / module))
if args.model == 'r-supcon':
    from dataset import ContrastiveClassificationDataset as Dataset
    from data_collators import DataCollatorContrastiveClassification
    from modeling import ContrastiveClassifierModel
else:
    from dataset import BaselineClassificationDataset as Dataset
weight = args.checkpoint / 'model.safetensors'
checkpoint_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
model = None
for path in args.inputs:
    data = Dataset(path, dataset_type='test', tokenizer=backbone, aug=False)
    if model is None:
        if args.model == 'r-supcon':
            state = load_file(str(weight))
            pos_neg = float(state['criterion.pos_weight'].item()) if 'criterion.pos_weight' in state else False
            model = ContrastiveClassifierModel(len_tokenizer=len(data.tokenizer), checkpoint_path=None, model=backbone, frozen=False, pos_neg=pos_neg)
            model.load_state_dict(state, strict=True)
            del state
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
            assert model.get_input_embeddings().weight.shape[0] == len(data.tokenizer)
    collator = DataCollatorContrastiveClassification(tokenizer=data.tokenizer) if args.model == 'r-supcon' else DataCollatorWithPadding(tokenizer=data.tokenizer)
    config = TrainingArguments(output_dir=str(args.output/'trainer'), per_device_eval_batch_size=8, fp16=True, report_to=[], disable_tqdm=True, dataloader_num_workers=0, seed=args.seed)
    trainer = Trainer(model=model, args=config, data_collator=collator)
    prediction = trainer.predict(data)
    values = prediction.predictions
    if isinstance(values, tuple):
        values = values[0]
    if args.model == 'r-supcon':
        probs = np.asarray(values).reshape(-1)
        predicted = (probs >= 0.5).astype(int)
    else:
        probs = torch.tensor(values).softmax(-1)[:, 1].numpy()
        predicted = np.argmax(values, axis=-1)
    frame = pd.read_pickle(path)
    labels = np.asarray(prediction.label_ids).reshape(-1).astype(int)
    assert labels.tolist() == frame['label'].tolist()
    stem = Path(path).name.removesuffix('.pkl.gz')
    pd.DataFrame({'pair_id': frame['pair_id'], 'label': labels, 'probability': probs, 'prediction': predicted}).to_csv(args.output/(stem+'_predictions.csv'), index=False)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    result = {'model': args.model, 'seed': args.seed, 'input': path, 'input_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash, 'pairs': len(labels), 'precision': precision, 'recall': recall, 'f1': f1}
    (args.output/(stem+'_metrics.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
(args.output/'complete.json').write_text(json.dumps({'inputs': args.inputs, 'checkpoint_sha256': checkpoint_hash})+'\n')
