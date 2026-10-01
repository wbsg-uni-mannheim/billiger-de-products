"""Generate independent classical seed jobs and feature prerequisites."""
import json
import os
import shlex
from pathlib import Path

root = Path('slurm_runs/v11/jobs')
root.mkdir(parents=True, exist_ok=True)
campaign = os.environ['V11_CAMPAIGN_DIR']  # isolated working copy of the repository on the cluster
records = []
for model in ('wordcooc', 'magellan'):
    environment = 'entitymatch' if model == 'magellan' else 'ditto-modern'
    prepare = f'''#!/bin/bash
#SBATCH --job-name=v11_features_{model}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_runs/logs/features_{model}_%j.out
#SBATCH --error=slurm_runs/logs/features_{model}_%j.err
set -euo pipefail
cd {campaign}
export BILLIGER_ENV=${{CONDA_ENVS:?}}/{environment}
source slurm_runs/env.sh
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -u -m src.processing.prepare_{model}
'''
    if model == 'magellan':
        prepare += 'python -u -m src.processing.prepare_cross_language_magellan\n'
        prepare += 'python -u src/v11/fix_magellan_metadata.py\n'
    (root/f'features_{model}.sh').write_text(prepare)
    cells = [(lang,family,size,False) for lang in ('de','en') for family in ('20cc80','50cc50','80cc20') for size in ('small','medium','large')]
    cells += [('de','80cc20','large',True)]
    for lang,family,size,cross in cells:
        for seed in (1,2,3):
            mode = 'cross' if cross else 'main'
            name = f'classical_{model}_{mode}_{lang}_{family}_{size}_{seed}'
            command = ['python','-u','src/v11/run_classical.py','--model',model,'--language',lang,'--family',family,'--size',size,'--seed',str(seed)] + (['--cross'] if cross else [])
            script = prepare[:prepare.index('python -u -m')] + shlex.join(command) + '\n'
            script = script.replace(f'v11_features_{model}',f'v11_{name}').replace(f'features_{model}_%j',name+'_%j').replace('--time=04:00:00','--time=48:00:00')
            (root/f'{name}.sh').write_text(script)
            records.append({'name':name,'script':f'slurm_runs/v11/jobs/{name}.sh','model':model,'seed':seed,'language':lang,'family':family,'size':size,'mode':mode})
Path('reports/v11_campaign_2026-09-06/cpu_jobs.json').write_text(json.dumps(records,indent=2)+'\n')
print('Generated',len(records),'classical seed jobs and two feature jobs')
