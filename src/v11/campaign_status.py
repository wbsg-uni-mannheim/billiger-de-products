"""Read Slurm and campaign artifacts without modifying jobs."""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
rows = []
for kind in ('gpu', 'cpu', 'gpt'):
    for row in json.loads((root / f'submitted_{kind}_jobs.json').read_text()):
        rows.append(dict(row, queue=kind))
ids = ','.join(row['job_id'] for row in rows)
raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ids, '-o', 'JobID,State,ExitCode,Elapsed,NodeList'], text=True)
states = {}
for line in raw.splitlines():
    job_id, state, exit_code, elapsed, nodes = line.split('|')[:5]
    states[job_id] = dict(state=state, exit_code=exit_code, elapsed=elapsed, nodes=nodes)
for row in rows:
    row.update(states.get(row['job_id'], dict(state='UNKNOWN')))
snapshot = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), total=len(rows),
                counts={kind: dict(Counter(r['state'] for r in rows if r['queue'] == kind)) for kind in ('gpu', 'cpu', 'gpt')},
                failures=[r for r in rows if r['state'] in ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'CANCELLED')],
                gpt_batches=[dict(path=str(p), **json.loads(p.read_text())) for p in root.glob('results/generated/gpt/**/batch_state.json')],
                gpt_missing_prediction_sets=[dict(path=str(p), **json.loads(p.read_text())) for p in root.glob('results/generated/gpt/**/partial_complete.json') if not (p.parent / 'complete.json').exists()],
                jobs=rows)
Path('campaign_status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
files = list(Path('src/v11').glob('*.py')) + [Path(p) for p in ('src/models/hiergat/train.py', 'src/models/ditto/train_ditto.py', 'src/models/ditto/ditto_light/ditto.py')]
Path('campaign_code_sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + '\n')
print(json.dumps({k:v for k,v in snapshot.items() if k not in ('jobs','gpt_batches')}, indent=2))
print('GPT batches:', Counter(r.get('status', 'submitted') for r in snapshot['gpt_batches']))
