"""Submit requested campaign jobs once and persist Slurm IDs immediately."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dependency', required=True)
parser.add_argument('--names', nargs='*')
args = parser.parse_args()
state = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', args.dependency, '-o', 'State'], text=True).strip()
assert state in ('COMPLETED', 'RUNNING', 'PENDING', 'COMPLETING'), state
dependency_args = [] if state == 'COMPLETED' else ['--dependency=afterok:' + args.dependency]
plan = json.loads(Path('gpu_jobs.json').read_text())
ledger = Path('submitted_gpu_jobs.json')
submitted = json.loads(ledger.read_text()) if ledger.exists() else []
known = {r['name'] for r in submitted}
for row in plan:
    if row['name'] in known or (args.names and row['name'] not in args.names):
        continue
    if 'checkpoint' in row:
        assert (Path(row['checkpoint'])/'model.safetensors').is_file(), row['checkpoint']
    job_id = subprocess.check_output(['sbatch', '--parsable', *dependency_args, row['script']], text=True).strip()
    record = dict(row, job_id=job_id)
    submitted.append(record)
    temporary = ledger.with_suffix('.tmp')
    temporary.write_text(json.dumps(submitted, indent=2)+'\n')
    temporary.replace(ledger)
    print(row['name'], job_id, flush=True)
