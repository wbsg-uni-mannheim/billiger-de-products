"""Prepare isolated v1.1 model inputs, retaining frozen normalized files exactly."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path('src/processing').resolve()))
from prepare_pairs import normalize_text, TEXT_LIMITS

root = Path.cwd()
base = root.parent
release = root / 'release'
manifest = json.loads((release / 'manifest.json').read_text())
assert manifest['ready_for_inference'] and not any(manifest['blockers'].values())
for relative, expected in manifest['candidate_sha256'].items():
    assert hashlib.sha256((release / relative).read_bytes()).hexdigest() == expected
for relative, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((base / 'data' / relative).read_bytes()).hexdigest() == expected
for language, processed in [('de', 'processed'), ('en', 'processed_en')]:
    raw = release / 'candidate_sets' / f'solute_{language}'
    link = root / 'data' / f'solute_{language}'
    if not link.exists():
        link.symlink_to(raw, target_is_directory=True)
    for source in sorted(raw.glob('*/*.json.gz')):
        folder = source.parent.name
        output_name = 'preprocessed_' + source.name.replace('.json.gz', '.pkl.gz')
        dest = root / 'data' / processed / folder / output_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        frozen = folder == 'training-sets' or (folder == 'validation-sets' and 'rnd000un' in source.name) or (folder == 'gold-standards_adjusted' and 'rnd100un' in source.name)
        if frozen:
            shutil.copy2(base / 'data' / processed / folder / output_name, dest)
            old = pd.read_pickle(dest)
            raw_df = pd.read_json(source, lines=True)
            assert old['pair_id'].tolist() == raw_df['pair_id'].tolist()
            assert old['label'].tolist() == raw_df['label'].tolist()
        else:
            pairs = pd.read_json(source, lines=True)
            for attribute, limit in TEXT_LIMITS.items():
                for side in ('left', 'right'):
                    col = f'{attribute}_{side}'
                    pairs[col] = pairs[col].map(lambda v: normalize_text(v, limit))
            pairs.reset_index(drop=True).to_pickle(dest, compression='gzip')
    print('Prepared', language, flush=True)
subprocess.run([sys.executable, '-m', 'src.processing.prepare_ditto_hiergat'], check=True)
# Preserve exact legacy serialized training and selection input, too.
for processed in ('processed', 'processed_en'):
    for model in ('ditto', 'hiergat'):
        folder = Path('data') / processed / model / 'data/final_output'
        for p in folder.glob('*.txt'):
            if '_train_' in p.name or ('rnd000un_valid' in p.name):
                shutil.copy2(base / p, p)
subprocess.run([sys.executable, '-m', 'src.processing.prepare_cross_language'], check=True)
(root / 'prepared.json').write_text(json.dumps({'ready': True, 'release_manifest_sha256': hashlib.sha256((release/'manifest.json').read_bytes()).hexdigest()}, indent=2)+'\n')
