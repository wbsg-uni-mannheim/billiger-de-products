"""Prepare, submit once, or inspect five bounded validation-based restarts."""
import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path

PLAN = Path('restart_jobs.json')
LEDGER = Path('submitted_restart_jobs.json')
EXPECTED = {'341196', '341248', '341282', '341283', '341289'}


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def prepare():
    if PLAN.exists():
        print('Existing frozen restart plan retained.')
        return
    audit = json.loads(Path('reports/collapse_validation/summary.json').read_text())
    eligible = {str(r['job_id']) for r in audit if r['positive_prediction_fraction'] in (0, 1)}
    assert eligible == EXPECTED
    jobs = json.loads(Path('campaign_status.json').read_text())['jobs']
    plan = []
    for old in jobs:
        if str(old['job_id']) not in eligible:
            continue
        assert old['state'] == 'COMPLETED'
        row = {k: old[k] for k in ('model', 'mode', 'language', 'family', 'size')}
        row.update(seed=3, original_seed=old['seed'], original_job_id=old['job_id'], original_output=old['output'], attempt=1)
        row['name'] = 'restart1_' + old['name'] + '_seed3'
        row['output'] = f"results/generated/restarts_v11_20260909/{old['model']}/{old['mode']}/{old['language']}/{old['family']}-{old['size']}/seed3"
        assert not Path(row['output']).exists()
        cmd = old['command'].copy()
        for flag, value in (('--run_id', '3'), ('--output_dir', row['output']), ('--logdir', row['output'] + '/checkpoints')):
            cmd[cmd.index(flag) + 1] = value
        row['command'] = cmd
        row['task'] = cmd[cmd.index('--task') + 1]
        config_file = cmd[cmd.index('--config_file') + 1]
        config = next(c for c in json.loads(Path(config_file).read_text()) if c['name'] == row['task'])
        row['validation_file'] = cmd[cmd.index('--validation_file') + 1] if '--validation_file' in cmd else config['validset']
        paths = list(Path('src/models/' + row['model']).rglob('*.py'))
        paths += [Path(config_file), Path(config['trainset']), Path(row['validation_file']), Path('src/v11/validate_restart.py')]
        row['frozen_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
        script = Path('slurm_runs/v11/restarts') / (row['name'] + '.sh')
        script.parent.mkdir(parents=True, exist_ok=True)
        lines = ['#!/bin/bash', '#SBATCH --job-name=' + row['name'], '#SBATCH --partition=gpu-vram-48gb,gpu-vram-94gb', '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=64G', '#SBATCH --time=48:00:00', '#SBATCH --output=slurm_runs/logs/' + row['name'] + '_%j.out', '#SBATCH --error=slurm_runs/logs/' + row['name'] + '_%j.err', 'set -euo pipefail', 'cd ' + shlex.quote(str(Path.cwd())), 'source slurm_runs/env.sh', 'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1', 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"', 'python src/v11/validate_restart.py --preflight --name ' + shlex.quote(row['name']), shlex.join(cmd), 'python -u src/v11/validate_restart.py --name ' + shlex.quote(row['name'])]
        script.write_text('\n'.join(lines) + '\n')
        subprocess.run(['bash', '-n', str(script)], check=True)
        row['script'] = str(script)
        plan.append(row)
    assert len(plan) == 5
    save(PLAN, plan)
    print('Prepared five independent jobs; original runs preserved.')


def submit():
    plan = json.loads(PLAN.read_text())
    submitted = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    known = {r['name'] for r in submitted}
    for row in plan:
        if row['name'] in known:
            continue
        subprocess.run(['python', 'src/v11/validate_restart.py', '--preflight', '--name', row['name']], check=True)
        job_id = subprocess.check_output(['sbatch', '--parsable', row['script']], text=True).strip()
        submitted.append(dict(row, job_id=job_id))
        save(LEDGER, submitted)
        print(row['name'], job_id, flush=True)


def status():
    ledger = json.loads(LEDGER.read_text())
    output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(str(r['job_id']) for r in ledger), '--format=JobID,State,Elapsed,ExitCode'], text=True)
    states = {parts[0]: parts[1:4] for line in output.splitlines() if (parts := line.split('|')) and len(parts) >= 4}
    results = []
    for row in ledger:
        decision_path = Path(row['output']) / 'restart_decision.json'
        results.append(dict(name=row['name'], job_id=row['job_id'], original_seed=row['original_seed'], replacement_seed=3,
                            state=states.get(str(row['job_id']), ['UNKNOWN'])[0],
                            decision=json.loads(decision_path.read_text()) if decision_path.exists() else None))
    save(Path('restart_status.json'), results)
    print(json.dumps(dict(counts=dict(Counter(r['state'] for r in results)), jobs=results), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'submit', 'status'])
    args = parser.parse_args()
    {'prepare': prepare, 'submit': submit, 'status': status}[args.action]()
