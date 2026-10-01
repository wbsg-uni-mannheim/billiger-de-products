"""Check restored pair coverage and metrics; resolve archived source pointers."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

restored = json.loads(Path('gpt_main_restoration.json').read_text())
assert len(restored) == 36
known_answers, checks = {}, []
for path in sorted(Path('recovered_gpt_history').glob('*/*.jsonl')):
    for row in map(json.loads, path.open()):
        known_answers.setdefault((row['request_sha256'], row['answer']), row['source'])
for path in sorted(Path('results/generated/gpt').glob('*/**/batch_results.jsonl')):
    requests_path = path.parent / 'new_requests.jsonl'
    state_path = path.parent / 'batch_state.json'
    if not requests_path.exists() or not state_path.exists():
        continue
    state = json.loads(state_path.read_text())
    all_requests = list(map(json.loads, (path.parent/'all_requests.jsonl').open()))
    responses = list(map(json.loads, path.open()))
    response_ids = {r['custom_id'] for r in responses}
    original_payload = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_requests if r['custom_id'] in response_ids)
    # Restore the original submitted input if an earlier offline worker wrote
    # an empty pending list over it. The saved pre-submission hash must match.
    assert hashlib.sha256(original_payload.encode()).hexdigest() == state['request_sha256'], path
    if hashlib.sha256(requests_path.read_bytes()).hexdigest() != state['request_sha256']:
        requests_path.write_text(original_payload)
    request_hashes = {r['custom_id']:hashlib.sha256(json.dumps(r['body'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for r in all_requests}
    batch_id = state['batch_id']
    for response in responses:
        if response.get('response', {}).get('status_code') == 200:
            answer = response['response']['body']['choices'][0]['message']['content']
            known_answers.setdefault((request_hashes[response['custom_id']], answer), batch_id)
for item in restored:
    out = Path(item['output'])
    inputs = {r['custom_id']:r for r in map(json.loads, (out/'all_requests.jsonl').open())}
    metadata = json.loads((out/'all_requests_meta.json').read_text())
    frame = pd.read_csv(out/'predictions.csv')
    assert len(frame) == len(inputs) and set(frame.pair_id) == set(inputs) == set(metadata)
    changed_sources = 0
    for index, row in frame.iterrows():
        request_hash = hashlib.sha256(json.dumps(inputs[row.pair_id]['body'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert request_hash == row.request_sha256
        assert int(row.label) == metadata[row.pair_id]['label']
        assert int(row.prediction) in (0, 1)
        source = str(row.source)
        # Replace derived CSV pointers with a verified raw answer for the exact
        # same request hash and answer, preserving reproducible source evidence.
        if source.startswith('results/generated/gpt/main/'):
            source = known_answers[(row.request_sha256, row.answer)]
        assert not source.startswith('results/generated/gpt/main/'), source
        if source != row.source:
            frame.loc[index, 'source'] = source
            changed_sources += 1
    if changed_sources:
        frame.to_csv(out/'predictions.csv', index=False)
    tp = int(((frame.label == 1) & (frame.prediction == 1)).sum())
    fp = int(((frame.label == 0) & (frame.prediction == 1)).sum())
    fn = int(((frame.label == 1) & (frame.prediction == 0)).sum())
    expected = {'precision':tp/(tp+fp) if tp+fp else 0.0, 'recall':tp/(tp+fn) if tp+fn else 0.0, 'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    metrics = json.loads((out/'metrics.json').read_text())
    assert all(math.isclose(metrics[k], v, abs_tol=1e-14) for k,v in expected.items())
    counts = json.loads((out/'request_counts.json').read_text())
    assert counts['new_requests'] == counts['unchanged_missing_predictions'] == 0
    assert (out/'complete.json').exists() and not (out/'partial_complete.json').exists()
    checks.append(dict(name=item['name'], pairs=len(frame), metrics_verified=True, missing_predictions=0,
                       sources_resolved=changed_sources, predictions_sha256=hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()))
Path('gpt_restoration_verification.json').write_text(json.dumps(checks, indent=2)+'\n')
print('Verified', len(checks), 'complete results;', sum(x['pairs'] for x in checks), 'pair predictions; zero missing predictions or restoration API requests')
