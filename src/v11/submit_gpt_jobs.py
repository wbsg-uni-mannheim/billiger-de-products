"""Submit GPT workers in four resumable lanes to bound simultaneous API batches."""
import json
import subprocess
from pathlib import Path

plan = json.loads(Path('gpt_jobs.json').read_text())
ledger = Path('submitted_gpt_jobs.json')
rows = json.loads(ledger.read_text()) if ledger.exists() else []
known = {r['name']: r for r in rows}
lanes = [None] * 4
for index, row in enumerate(plan):
    lane = index % len(lanes)
    if row['name'] in known:
        lanes[lane] = known[row['name']]['job_id']
        continue
    dependency = []
    if lanes[lane]:
        state = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', lanes[lane], '-o', 'State'], text=True).strip()
        if state in ('RUNNING', 'PENDING', 'COMPLETING'):
            dependency = ['--dependency=afterany:' + lanes[lane]]
        else:
            assert state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'), state
    job_id = subprocess.check_output(['sbatch', '--parsable', *dependency, row['script']], text=True).strip()
    rows.append(dict(row, job_id=job_id, lane=lane))
    temporary = ledger.with_suffix('.tmp')
    temporary.write_text(json.dumps(rows, indent=2) + '\n')
    temporary.replace(ledger)
    lanes[lane] = job_id
    print(row['name'], job_id, flush=True)
