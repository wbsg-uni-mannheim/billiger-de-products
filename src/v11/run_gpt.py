"""Run one GPT prompt/split/replicate; reuse only exact historical request bodies."""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('main','cross'), required=True)
parser.add_argument('--language', default='de')
parser.add_argument('--family', default='80cc20')
parser.add_argument('--condition', default='050')
parser.add_argument('--variant', default='de_de')
parser.add_argument('--prompt', choices=('simple','rule_guided'), required=True)
parser.add_argument('--replicate', type=int, default=1)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--refresh', action='store_true')
parser.add_argument('--restore-only', action='store_true')
a = parser.parse_args()
base = Path.cwd().parent
load_dotenv(base / '.env')
assert os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY missing'
client = OpenAI(max_retries=2)
name = f'{a.language}_{a.family}_{a.condition}' if a.mode == 'main' else a.variant
out = Path('results/generated/gpt') / a.mode / a.prompt / name / str(a.replicate)
out.mkdir(parents=True, exist_ok=True)
if (out/'complete.json').exists() and not a.refresh:
    raise SystemExit('Already complete')


def body_hash(req):
    return hashlib.sha256(json.dumps(req['body'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def response_answer(response):
    try:
        return response['response']['body']['choices'][0]['message']['content']
    except (KeyError,TypeError,IndexError):
        return None


def parse_answer(answer):
    text = str(answer).strip().lower()
    if text in ('ja','yes','1') or text.startswith(('ja.','ja,','yes.','yes,')):
        return 1
    if text in ('nein','no','0') or text.startswith(('nein.','nein,','no.','no,')):
        return 0
    return -1


batch_path = out/'all_requests.jsonl'
if a.mode == 'main':
    language_name = 'german' if a.language == 'de' else 'english'
    filename = f'gpt_batch_{language_name}' + ('_new_prompt' if a.prompt=='rule_guided' else '') + '.py'
    spec = importlib.util.spec_from_file_location('gpt_builder','src/models/gpt/'+filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.args = SimpleNamespace(gptmodel='gpt-5.2')
    module.build_batch_file(a.family,a.condition,str(batch_path))
else:
    from src.models.gpt.gpt_batch_cross_language import build_batch_file
    build_batch_file(a.variant,a.prompt,'gpt-5.2',batch_path)
metadata = json.loads(batch_path.with_name('all_requests_meta.json').read_text())
requests = [json.loads(line) for line in batch_path.open()]
assert len(requests) == len(metadata)
# Unchanged main-table prompts never trigger a paid fallback when their old
# predictions are unavailable. Gold-label changes do not change this hash.
original_hashes = set()
if a.mode == 'main':
    original_path = (out / 'original_requests.jsonl').resolve()
    campaign_cwd = Path.cwd()
    try:
        os.chdir(base)
        module.build_batch_file(a.family, a.condition, str(original_path))
    finally:
        os.chdir(campaign_cwd)
    original_hashes = {body_hash(req) for req in map(json.loads, original_path.open())}
# Historical cross-language replicates remain separate stochastic draws.
cache = {}


def load_recovered(path):
    if path.exists():
        for row in map(json.loads, path.open()):
            assert parse_answer(row['answer']) in (0, 1)
            cache.setdefault(row['request_sha256'], {'answer':row['answer'], 'source':row['source']})


recovered_root = Path('recovered_gpt_history') / a.prompt
if a.mode == 'main' and a.replicate == 1:
    # Restore the original draw for this exact published cell before falling
    # back to other sets with the same request body.
    load_recovered(recovered_root / (name + '.jsonl'))
source_dir = base / 'data/batch_inputs/cross_language/gpt/gpt-5.2' / a.prompt
for original in sorted(source_dir.glob('*.jsonl')):
    rep = 3 if original.stem.endswith('_run3') else 2 if original.stem.endswith('_run2') else 1
    if rep != a.replicate:
        continue
    old_inputs = {r['custom_id']:r for r in map(json.loads,original.open())}
    if a.mode == 'cross':
        original_hashes.update(body_hash(req) for req in old_inputs.values())
    result_path = Path(str(original).replace('/batch_inputs/','/batch_results/'))
    if not result_path.exists():
        continue
    for response in map(json.loads,result_path.open()):
        req = old_inputs.get(response.get('custom_id'))
        answer = response_answer(response)
        if req and answer is not None and parse_answer(answer) in (0,1):
            cache.setdefault(body_hash(req), {'answer':answer,'source':str(result_path)})
# Reuse answers already obtained in this campaign as well. Keep repetitions
# separate, but permit the same request to be shared by overlapping test sets.
for mode in ('main', 'cross'):
    for prediction_file in sorted((Path('results/generated/gpt') / mode / a.prompt).glob(f'*/{a.replicate}/predictions.csv')):
        if not any((prediction_file.parent / marker).exists() for marker in ('complete.json', 'partial_complete.json')):
            continue
        for row in pd.read_csv(prediction_file).to_dict(orient='records'):
            if parse_answer(row['answer']) in (0, 1):
                cache.setdefault(row['request_sha256'], {'answer':row['answer'], 'source':row['source']})
if a.replicate == 1:
    for recovered_file in sorted(recovered_root.glob('*.jsonl')):
        load_recovered(recovered_file)
# Every new request has its own custom ID; repeated prompts are not treated as
# additional independent replicates by copying another new response.
answers, pending, unchanged_missing = {}, [], []
for req in requests:
    key = body_hash(req)
    if key in cache:
        answers[req['custom_id']] = dict(cache[key],request_sha256=key,reused=True)
    elif key in original_hashes:
        unchanged_missing.append(req['custom_id'])
    else:
        pending.append(req)
counts = {'total':len(requests),'cached':len(answers),'new_requests':len(pending),'unchanged_missing_predictions':len(unchanged_missing),'replicate':a.replicate}
(out/'request_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
(out/'unchanged_missing_predictions.json').write_text(json.dumps(unchanged_missing)+'\n')
print('GPT request counts',counts,flush=True)
if a.restore_only:
    assert not pending and not unchanged_missing, f'Offline restoration incomplete: {counts}'
if a.prepare_only:
    raise SystemExit(0)
missing_path = out/'new_requests.jsonl'
state_path = out/'batch_state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}
if pending:
    payload = ''.join(json.dumps(req,ensure_ascii=False)+'\n' for req in pending)
    request_hash = hashlib.sha256(payload.encode()).hexdigest()
    assert state.get('request_sha256',request_hash) == request_hash
    missing_path.write_text(payload)
    if not state.get('batch_id'):
        if not state.get('input_file_id'):
            with missing_path.open('rb') as stream:
                uploaded = client.files.create(file=stream,purpose='batch')
            state.update(input_file_id=uploaded.id,request_sha256=request_hash)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
        # Stable idempotency key prevents duplicate paid submissions on retry.
        idempotency = hashlib.sha256((str(out.resolve())+request_hash).encode()).hexdigest()
        while True:
            try:
                batch = client.batches.create(input_file_id=state['input_file_id'],endpoint='/v1/chat/completions',completion_window='24h',extra_headers={'Idempotency-Key':idempotency})
                break
            except RateLimitError:
                print('Batch enqueue limit; retry in 30 seconds',flush=True)
                time.sleep(30)
        state['batch_id'] = batch.id
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print('Submitted',batch.id,flush=True)
    while True:
        batch = client.batches.retrieve(state['batch_id'])
        state.update(status=batch.status,request_counts=batch.request_counts.model_dump() if batch.request_counts else None)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        if batch.status == 'completed':
            break
        if batch.status in ('failed','expired','cancelled'):
            if batch.error_file_id:
                (out/'batch_errors.jsonl').write_bytes(client.files.content(batch.error_file_id).read())
            raise RuntimeError(f'Batch {batch.id} ended {batch.status}: {batch.errors}')
        time.sleep(30)
    result_path = out/'batch_results.jsonl'
    result_path.write_bytes(client.files.content(batch.output_file_id).read())
    for response in map(json.loads,result_path.open()):
        answer = response_answer(response)
        if answer is not None:
            answers[response['custom_id']] = {'answer':answer,'source':state['batch_id'],'reused':False}
    if batch.error_file_id:
        (out/'batch_errors.jsonl').write_bytes(client.files.content(batch.error_file_id).read())
assert set(answers) == set(metadata) - set(unchanged_missing), 'Missing/error responses; do not score an incomplete split'
rows = []
for req in requests:
    pid = req['custom_id']
    if pid not in answers:
        continue
    answer = answers[pid]
    rows.append({'pair_id':pid,'label':metadata[pid]['label'],'prediction':parse_answer(answer['answer']),**answer,'request_sha256':body_hash(req)})
frame = pd.DataFrame(rows, columns=['pair_id','label','prediction','answer','source','reused','request_sha256'])
frame.to_csv(out/'predictions.csv',index=False)
assert (frame.prediction >= 0).all(), 'Invalid answers; inspect before scoring'
if unchanged_missing:
    (out/'partial_complete.json').write_text(json.dumps(counts,indent=2)+'\n')
    print('Changed-input inference complete; full scoring needs the missing historical predictions. No full-split metric written.', flush=True)
    raise SystemExit(0)
precision,recall,f1,_ = precision_recall_fscore_support(frame.label,frame.prediction,average='binary',zero_division=0)
metrics = {'pairs':len(frame),'precision':precision,'recall':recall,'f1':f1,'replicate':a.replicate,'cached':int(frame.reused.sum()),'new_requests':len(pending)}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
(out/'complete.json').write_text(json.dumps(metrics)+'\n')
(out/'partial_complete.json').unlink(missing_ok=True)
print(json.dumps(metrics),flush=True)
