"""Audit held GPT jobs without submitting API requests."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

held = set(json.loads(Path('gpt_cost_review_holds.json').read_text()))
rows = [r for r in json.loads(Path('submitted_gpt_jobs.json').read_text()) if r['job_id'] in held]


def prepare(row):
    args = [sys.executable, 'src/v11/run_gpt.py', '--prepare-only']
    for key in ('mode', 'language', 'family', 'condition', 'variant', 'prompt', 'replicate'):
        if key in row:
            args += ['--' + key, str(row[key])]
    result = subprocess.run(args, text=True, capture_output=True)
    log = Path('slurm_runs/logs') / (row['name'] + '_delta_audit.log')
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (row['name'], str(log))
    name = f"{row['language']}_{row['family']}_{row['condition']}" if row['mode'] == 'main' else row['variant']
    out = Path('results/generated/gpt') / row['mode'] / row['prompt'] / name / str(row['replicate'])
    return dict(name=row['name'], job_id=row['job_id'], mode=row['mode'], prompt=row['prompt'], **json.loads((out / 'request_counts.json').read_text()))


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(prepare, rows))
Path('gpt_delta_cost_audit.json').write_text(json.dumps(results, indent=2) + '\n')
for mode in ('main', 'cross'):
    selected = [r for r in results if r['mode'] == mode]
    print(mode, 'jobs', len(selected), {k:sum(r[k] for r in selected) for k in ('total', 'cached', 'new_requests', 'unchanged_missing_predictions')}, flush=True)
