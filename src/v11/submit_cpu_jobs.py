"""Submit one classical seed per job after its feature preparation succeeds."""
import argparse
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--wordcooc-dependency',required=True)
p.add_argument('--magellan-dependency',required=True)
p.add_argument('--names',nargs='*')
a=p.parse_args()
dependencies = {}
for model, job_id in (('wordcooc', a.wordcooc_dependency), ('magellan', a.magellan_dependency)):
    state = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job_id, '-o', 'State'], text=True).strip()
    assert state in ('COMPLETED', 'RUNNING', 'PENDING', 'COMPLETING'), state
    dependencies[model] = [] if state == 'COMPLETED' else ['--dependency=afterok:' + job_id]
plan=json.loads(Path('cpu_jobs.json').read_text())
ledger=Path('submitted_cpu_jobs.json')
rows=json.loads(ledger.read_text()) if ledger.exists() else []
known={r['name'] for r in rows}
for r in plan:
    if r['name'] in known or (a.names and r['name'] not in a.names):
        continue
    jid=subprocess.check_output(['sbatch','--parsable',*dependencies[r['model']],r['script']],text=True).strip()
    rows.append(dict(r,job_id=jid))
    tmp=ledger.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows,indent=2)+'\n')
    tmp.replace(ledger)
    print(r['name'],jid,flush=True)
