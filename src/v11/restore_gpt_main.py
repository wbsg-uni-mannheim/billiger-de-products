"""Restore all main-table metrics offline from verified old and new answers."""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

manifest = json.loads(Path('recovered_gpt_history/manifest.json').read_text())
assert len(manifest) == 36 and all(row['published_f1_matches'] for row in manifest)
jobs = [r for r in json.loads(Path('gpt_jobs.json').read_text()) if r['mode'] == 'main']
jobs += [dict(name=f'gpt_{prompt}_main_{language}_{family}_100_1', mode='main', language=language,
              family=family, condition='100', prompt=prompt, replicate=1)
         for prompt in ('simple', 'rule_guided') for language in ('de', 'en') for family in ('20cc80', '50cc50', '80cc20')]
report = Path('gpt_main_restoration.json')
done = {r['name']:r for r in json.loads(report.read_text())} if report.exists() else {}


def restore(row):
    if row['name'] in done:
        return done[row['name']]
    cell = f"{row['language']}_{row['family']}_{row['condition']}"
    out = Path('results/generated/gpt/main') / row['prompt'] / cell / '1'
    archive = Path('results/generated/gpt/main_before_history_recovery') / row['prompt'] / cell / '1'
    assert not archive.exists(), archive
    archive.mkdir(parents=True)
    for name in ('predictions.csv','metrics.json','complete.json','partial_complete.json','request_counts.json','unchanged_missing_predictions.json'):
        if (out/name).exists():
            shutil.copy2(out/name, archive/name)
    args = [sys.executable, 'src/v11/run_gpt.py', '--refresh', '--restore-only']
    for key in ('mode','language','family','condition','prompt','replicate'):
        args += ['--'+key, str(row[key])]
    result = subprocess.run(args, text=True, capture_output=True)
    log = Path('slurm_runs/logs') / (row['name'] + '_history_restore.log')
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (row['name'], str(log))
    counts = json.loads((out/'request_counts.json').read_text())
    assert counts['new_requests'] == 0 and counts['unchanged_missing_predictions'] == 0
    return dict(name=row['name'], output=str(out), **json.loads((out/'metrics.json').read_text()))


with ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(restore, jobs))
Path('gpt_main_restoration.json').write_text(json.dumps(rows, indent=2)+'\n')
print('Restored', len(rows), 'complete main-table results without any API requests', flush=True)
