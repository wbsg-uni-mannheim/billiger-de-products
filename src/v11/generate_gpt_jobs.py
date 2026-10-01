"""Create one GPT job per prompt/split/repetition with independent output paths."""
import json
import os
import shlex
from pathlib import Path

campaign = os.environ['V11_CAMPAIGN_DIR']  # isolated working copy of the repository on the cluster
rows = []
for prompt in ('simple', 'rule_guided'):
    cases = [dict(mode='main', language=lang, family=fam, condition=cond, replicate=1)
             for lang in ('de', 'en') for fam in ('20cc80', '50cc50', '80cc20') for cond in ('000', '050')]
    cases += [dict(mode='cross', variant=variant, replicate=rep)
              for variant in ('de_de', 'de_en', 'en_de', 'en_en', 'random') for rep in (1, 2, 3)]
    for case in cases:
        name = 'gpt_' + prompt + '_' + '_'.join(str(v) for v in case.values())
        command = ['python', '-u', 'src/v11/run_gpt.py', '--prompt', prompt]
        for key, value in case.items():
            command += ['--' + key, str(value)]
        script = f'''#!/bin/bash
#SBATCH --job-name=v11_{name}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_runs/logs/{name}_%j.out
#SBATCH --error=slurm_runs/logs/{name}_%j.err
set -euo pipefail
cd {campaign}
source slurm_runs/env.sh
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{shlex.join(command)}
'''
        path = Path('slurm_runs/v11/jobs') / (name + '.sh')
        path.write_text(script)
        rows.append(dict(name=name, script=str(path), prompt=prompt, **case))
Path('reports/v11_campaign_2026-09-06/gpt_jobs.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Generated', len(rows), 'GPT jobs')
