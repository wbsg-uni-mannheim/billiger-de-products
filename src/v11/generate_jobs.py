"""Generate one isolated GPU job per model/cell/language/seed."""
import json
import os
import shlex
from pathlib import Path

BASE = os.environ['BILLIGER_BASE_DIR']  # repository checkout holding the v1.0 runs and checkpoints
CAMPAIGN = BASE + '/v11-reruns-20260906'
OUT = Path('slurm_runs/v11/jobs')
OUT.mkdir(parents=True, exist_ok=True)
records = []


def add(record, command):
    name = '_'.join(str(record[k]) for k in ('kind', 'model', 'mode', 'language', 'family', 'size', 'seed'))
    record['name'] = name
    record['script'] = f'slurm_runs/v11/jobs/{name}.sh'
    record['command'] = command
    walltime = '02:00:00' if record['kind'] == 'infer' else '48:00:00'
    script = f'''#!/bin/bash
#SBATCH --job-name=v11_{name}
#SBATCH --partition=gpu-vram-48gb,gpu-vram-94gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time={walltime}
#SBATCH --output=slurm_runs/logs/{name}_%j.out
#SBATCH --error=slurm_runs/logs/{name}_%j.err
set -euo pipefail
cd {shlex.quote(CAMPAIGN)}
source slurm_runs/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${{PYTHONPATH:+:$PYTHONPATH}}"
test -f prepared.json
python -c 'import torch; assert torch.cuda.is_available(), "No GPU allocated"'
{shlex.join(command)}
'''
    (OUT / (name + '.sh')).write_text(script)
    records.append(record)


for model in ('hiergat', 'ditto'):
    cells = [(lang, family, size, 'main') for size in ('small', 'medium', 'large')
             for lang in ('de', 'en') for family in ('20cc80', '50cc50', '80cc20')]
    cells += [('de', '80cc20', 'large', 'cross')]
    for language, family, size, mode in cells:
        for seed in (0, 1, 2):
            cell = f'{family}-{size}'
            output = f'results/generated/{model}/{mode}/{language}/{cell}/{seed}'
            config_stem = 'task' if model == 'hiergat' else 'configs'
            config = f'src/models/{model}/{config_stem}' + ('_en' if language == 'en' else '') + '.json'
            entry = 'train.py' if model == 'hiergat' else 'train_ditto.py'
            command = ['python', '-u', f'src/models/{model}/{entry}', '--task', f'final_{size}_{family}rnd000un', '--run_id', str(seed), '--batch_size', '16' if model == 'hiergat' else '64', '--max_len', '256', '--lr', '5e-6' if model == 'hiergat' else '5e-5', '--n_epochs', '50', '--finetuning', '--save_model', '--logdir', output + '/checkpoints', '--output_dir', output, '--lm', 'roberta', '--config_file', config]
            command += ['--split'] if model == 'hiergat' else ['--da', 'del']
            if mode == 'cross':
                command += ['--validation_file', f'data/processed/{model}/data/final_output/preprocessed_products80cc20rnd000un_valid_large.txt', '--cross_language_test_dir', f'data/processed_cross_language/{model}/data/final_output']
            add(dict(kind='train', model=model, mode=mode, language=language, family=family, size=size, seed=seed, output=output), command)

inventory = json.loads(Path('reports/v11_checkpoint_inventory_2026-09-06.json').read_text())
for model in ('roberta', 'r-supcon', 'xlmr'):
    for item in inventory[model]['main_weights']:
        language, family, size, seed = item['language'], item['cc'] + 'cc' + str(100-int(item['cc'])), item['size'], int(item['seed'])
        processed = 'processed_en' if language == 'en' else 'processed'
        inputs = [f'data/{processed}/gold-standards_adjusted/preprocessed_products{family}rnd{condition}un_gs.pkl.gz' for condition in ('000', '050', '100')]
        inputs += [f'data/{processed}/validation-sets/preprocessed_products{family}rnd{condition}un_valid_{size}.pkl.gz' for condition in ('000', '050', '100')]
        checkpoint = str(Path(item['path']).parent)
        output = f'results/generated/{model}/main/{language}/{family}-{size}/{seed}'
        command = ['python', '-u', 'src/v11/infer_transformer.py', '--model', model, '--checkpoint', checkpoint, '--seed', str(seed), '--output', output, '--inputs', *inputs]
        add(dict(kind='infer', model=model, mode='main', language=language, family=family, size=size, seed=seed, checkpoint=checkpoint, output=output), command)
    for seed in (0, 1, 2):
        checkpoint = BASE + f'/results/generated/cross_language/{model}/80cc20-large/{seed}'
        output = f'results/generated/{model}/cross/de/80cc20-large/{seed}'
        inputs = [f'data/processed_cross_language/gold-standards_adjusted/preprocessed_products80cc20rnd050un_gs_{variant}.pkl.gz' for variant in ('de_de', 'de_en', 'en_de', 'en_en', 'random')]
        command = ['python', '-u', 'src/v11/infer_transformer.py', '--model', model, '--checkpoint', checkpoint, '--seed', str(seed), '--output', output, '--inputs', *inputs]
        add(dict(kind='infer', model=model, mode='cross', language='de', family='80cc20', size='large', seed=seed, checkpoint=checkpoint, output=output), command)
Path('reports/v11_campaign_2026-09-06/gpu_jobs.json').write_text(json.dumps(records, indent=2)+'\n')
print('Generated', len(records), 'jobs:', sum(r['kind']=='train' for r in records), 'train,', sum(r['kind']=='infer' for r in records), 'inference')
