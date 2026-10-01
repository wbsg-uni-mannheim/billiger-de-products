"""Accept a bounded restart using only its frozen Seen validation split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def check_files(row):
    for name, expected in row['frozen_sha256'].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source/input changed: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    row = next(r for r in json.loads(Path('restart_jobs.json').read_text()) if r['name'] == args.name)
    check_files(row)
    if args.preflight:
        print('Frozen training/validation inputs and code verified.', flush=True)
        return
    checkpoint = next(Path(row['output']).glob('checkpoints/**/model.pt'))
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    hp = ck['args']
    assert hp['run_id'] == row['seed'] == 3
    assert hp['task'] == row['task']
    config = next(c for c in json.loads(Path(hp['config_file']).read_text()) if c['name'] == hp['task'])
    validation_path = hp.get('validation_file') or config['validset']
    assert validation_path == row['validation_file']
    if row['model'] == 'ditto':
        sys.path.insert(0, str(Path('src/models/ditto').resolve()))
        from ditto_light.dataset import DittoDataset
        from ditto_light.ditto import DittoModel
        dataset = DittoDataset(validation_path, lm=hp['lm'])
        model = DittoModel(device='cuda', lm=hp['lm'], alpha_aug=hp['alpha_aug'])
        batch_size = hp['batch_size'] * 16
        threshold = float(ck['threshold'])
    else:
        sys.path.insert(0, str(Path('src/models/hiergat').resolve()))
        from model.dataset import Dataset
        from model.model import TranHGAT
        dataset = Dataset(validation_path, config['category'], lm=hp['lm'], lm_path=hp['lm_path'], split=hp['split'])
        model = TranHGAT(dataset.get_attr_num(), 'cuda', hp['finetuning'], lm=hp['lm'], lm_path=hp['lm_path'])
        batch_size = hp['batch_size']
        threshold = None
    model.load_state_dict(ck['model'], strict=True)
    model.cuda().eval()
    labels, probabilities, predictions = [], [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.pad):
            if row['model'] == 'ditto':
                x, y = batch
                logits = model(x)
                prob = logits.softmax(-1)[:, 1]
                pred = prob > threshold
            else:
                _, x, y, _, mask = batch
                logits, _, pred = model(x, y, mask)
                prob = logits.softmax(-1)[:, 1]
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite validation logits; restart cannot be accepted.')
            labels.extend(y.tolist())
            probabilities.extend(prob.cpu().tolist())
            predictions.extend(pred.cpu().tolist())
    assert len(labels) == len(dataset) and set(labels) == {0, 1}
    validation_f1 = float(f1_score(labels, predictions, zero_division=0))
    if abs(validation_f1 - float(ck['validation_f1'])) > 1e-8:
        raise RuntimeError('Checkpoint validation score does not reproduce; manual investigation required.')
    accepted = len(set(predictions)) == 2
    result = dict(name=row['name'], original_job_id=row['original_job_id'], original_seed=row['original_seed'], replacement_seed=row['seed'], attempt=1,
                  accepted=accepted, reason='nonconstant_seen_validation' if accepted else 'constant_seen_validation_no_more_retries',
                  validation_file=validation_path, validation_f1=validation_f1, positive_prediction_fraction=float(np.mean(predictions)),
                  pairs=len(labels), checkpoint=str(checkpoint), checkpoint_epoch=ck['epoch'], threshold=threshold,
                  selected_output=row['output'] if accepted else row['original_output'], original_output=row['original_output'])
    out = Path(row['output'])
    np.savez_compressed(out / 'restart_validation.npz', labels=labels, probability=probabilities, prediction=predictions)
    temporary = out / 'restart_decision.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(out / 'restart_decision.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
