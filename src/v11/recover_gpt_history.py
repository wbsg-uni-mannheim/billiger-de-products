"""Verify the original thesis GPT batches and create exact-request caches."""
import csv
import hashlib
import json
import os
import math
import re
from pathlib import Path

BACKUP = Path(os.environ['THESIS_DIR'])  # checkout of the original thesis code with its GPT batch results
CURRENT = Path.cwd().parent
OUT = Path('recovered_gpt_history')
OUT.mkdir(exist_ok=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_hash(request):
    return hashlib.sha256(json.dumps(request['body'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


manifest = []
for language, word in (('de', 'german'), ('en', 'english')):
    folder = BACKUP / f'data/batch_results/gpt_{language}/gpt-5.2'
    for response_path in sorted(folder.glob('*.jsonl')):
        match = re.fullmatch(r'gpt-5\.2_(\d+cc\d+)_(\d{3})_batched_' + word + r'_(easy_prompt|hard_prompt|gard_prompt|ard_prompt)\.jsonl', response_path.name)
        assert match, response_path
        family, condition, suffix = match.groups()
        prompt = 'simple' if suffix == 'easy_prompt' else 'rule_guided'
        canonical = 'easy_prompt' if prompt == 'simple' else 'hard_prompt'
        request_path = BACKUP / f'data/batch_inputs/gpt_{language}/gpt-5.2' / response_path.name
        metadata_path = request_path.with_name(request_path.stem + '_meta.json')
        export_path = BACKUP / f'src/models/gpt/reports_{language}/gpt-5.2/csv_results/products_{family}_{condition}un_batched_{word}_{canonical}.csv'
        reference_path = CURRENT / f'src/models/gpt/reports_{language}/gpt-5.2/f1/f1_score_{family}_{condition}un_{word}_{canonical}.txt'
        requests = {r['custom_id']:r for r in map(json.loads, request_path.open())}
        responses = {r['custom_id']:r for r in map(json.loads, response_path.open())}
        metadata = json.loads(metadata_path.read_text())
        exports = {r['Pair_ID']:r for r in csv.DictReader(export_path.open())}
        assert set(requests) == set(metadata)
        assert set(responses) == set(exports) and set(responses) <= set(requests)
        missing_original_responses = sorted(set(requests) - set(responses))
        cached, tp, fp, fn = [], 0, 0, 0
        for pair_id, response in responses.items():
            request = requests[pair_id]
            assert request['body']['model'] == 'gpt-5.2'
            assert response['response']['status_code'] == 200 and not response.get('error')
            body = response['response']['body']
            assert body['model'].startswith('gpt-5.2')
            answer = body['choices'][0]['message']['content']
            normalized = answer.strip().lower().rstrip('.,')
            assert normalized in ('ja', 'nein', 'yes', 'no'), (response_path, pair_id, answer)
            predicted = int(normalized in ('ja', 'yes'))
            label = int(metadata[pair_id]['label'])
            assert predicted == int(exports[pair_id]['Answer_binary'])
            assert label == int(exports[pair_id]['Label'])
            tp += predicted == 1 and label == 1
            fp += predicted == 1 and label == 0
            fn += predicted == 0 and label == 1
            cached.append(dict(request_sha256=body_hash(request), answer=answer, source=str(response_path), original_pair_id=pair_id))
        f1 = 2 * tp / (2 * tp + fp + fn)
        reference = float(re.search(r'F1 Score:\s*([0-9.eE+-]+)', reference_path.read_text()).group(1))
        target = OUT / prompt / f'{language}_{family}_{condition}.jsonl'
        target.parent.mkdir(exist_ok=True)
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in cached)
        if target.exists():
            # Cache row order follows the input file for stable resumability.
            existing = {r['original_pair_id']:r for r in map(json.loads, target.open())}
            assert existing == {r['original_pair_id']:r for r in cached}, target
        else:
            target.write_text(content)
        record = dict(language=language, family=family, condition=condition, prompt=prompt, pairs=len(cached), cache=str(target),
                      missing_original_responses=missing_original_responses, f1_from_original_responses=f1, published_f1=reference, published_f1_matches=math.isclose(f1, reference, abs_tol=1e-14),
                      sources={str(p):digest(p) for p in (request_path, response_path, metadata_path, export_path, reference_path)}, cache_sha256=digest(target))
        manifest.append(record)
        print(language, family, condition, prompt, len(cached), 'published F1 matches', record['published_f1_matches'], flush=True)
assert len(manifest) == 36
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Recovered', len(manifest), 'original batch sets;', sum(r['pairs'] for r in manifest), 'answers;', sum(r['published_f1_matches'] for r in manifest), 'published F1 values reproduced', flush=True)
