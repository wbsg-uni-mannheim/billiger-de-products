"""Refit one classical seed using historical validation-selected parameters."""
import argparse
import ast
import importlib
import inspect
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('wordcooc', 'magellan'), required=True)
parser.add_argument('--language', choices=('de', 'en'), required=True)
parser.add_argument('--family', required=True)
parser.add_argument('--size', required=True)
parser.add_argument('--seed', type=int, choices=(1, 2, 3), required=True)
parser.add_argument('--cross', action='store_true')
a = parser.parse_args()
benchmark = importlib.import_module(f'src.models.{a.model}.run_{a.model}')
function_name = 'run_' + a.model
# The only loop change is exposing the existing 1,2,3 seed loop to this job.
code = inspect.getsource(getattr(benchmark, function_name))
assert code.count('for run in range(1, 4):') == 1
code = code.replace('for run in range(1, 4):', f'for run in ({a.seed},):')
exec(compile(code, inspect.getfile(benchmark), 'exec'), benchmark.__dict__)
original_search = benchmark.RandomizedSearchCV
base = Path.cwd().parent
mode = 'cross' if a.cross else 'main'
output = Path(f'results/generated/{a.model}/{mode}/{a.language}/{a.family}-{a.size}/{a.seed}')
output.mkdir(parents=True, exist_ok=True)
benchmark.RESULT_ROOT = str(output)
processed = 'processed' if a.language == 'de' else 'processed_en'
variant = f'products{a.family}rnd000un'
if a.cross:
    from src.cross_language.common import EXPERIMENT_NAME
    experiment = EXPERIMENT_NAME
    root = Path('data/processed_cross_language') / a.model
    if a.model == 'wordcooc':
        train = root / f'preprocessed_{variant}_train_large_wordcooc.pkl.gz'
        valid = root / f'preprocessed_{variant}_valid_large_wordcooc.pkl.gz'
        tests = [root / f'preprocessed_products80cc20rnd050un_gs_{v}_wordcooc.pkl.gz' for v in ('de_de','de_en','en_de','en_en','random')]
    else:
        train = root / f'preprocessed_{variant}_train_large_cross_magellan_pairs_formatted.csv'
        valid = root / f'preprocessed_{variant}_valid_large_cross_magellan_pairs_formatted.csv'
        tests = [root / f'preprocessed_products80cc20rnd050un_gs_{v}_magellan_pairs_formatted.csv' for v in ('de_de','de_en','en_de','en_en','random')]
    report_root = base / 'results/generated/cross_language' / a.model / experiment
else:
    experiment = 'learning-curve'
    root = Path('data') / processed / a.model / 'learning-curve'
    if a.model == 'wordcooc':
        train = root / f'preprocessed_{variant}_train_{a.size}_wordcooc.pkl.gz'
        valid = root / train.name.replace('train_', 'valid_')
        tests = [root / (train.name.replace('.pkl.gz', '') + f'_preprocessed_products{a.family}rnd{c}un_gs.pkl.gz') for c in ('000','050','100')]
    else:
        root /= 'formatted'
        train = root / f'preprocessed_{variant}_train_{a.size}_magellan_pairs_formatted.csv'
        valid = root / train.name.replace('train_', 'valid_')
        tests = [root / f'preprocessed_products{a.family}rnd{c}un_gs_magellan_pairs_formatted.csv' for c in ('000','050','100')]
    report_root = base / 'results/generated' / a.model / a.language / experiment
provenance = []
for test in tests:
    suffix = '.pkl.gz' if a.model == 'wordcooc' else '.csv'
    report = report_root / (train.name.removesuffix(suffix) + '_' + test.name.removesuffix(suffix) + '.csv')
    rows = pd.read_csv(report, sep='#####', engine='python') if report.exists() else pd.DataFrame()
    classifier_names = {type(v['clf']).__name__: name for name,v in benchmark.classifiers.items()}

    def make_search(**kwargs):
        name = classifier_names[type(kwargs['estimator']).__name__]
        matches = rows[rows['model'] == name] if not rows.empty else rows
        if len(matches) < a.seed:
            provenance.append({'report':str(report),'model':name,'seed':a.seed,'selection':'new_search_missing_historical_row'})
            return original_search(**kwargs)
        selected = matches.iloc[a.seed-1]
        params = ast.literal_eval(selected['best_params'])
        provenance.append({'report':str(report),'model':name,'seed':a.seed,'selection':'reuse_frozen_validation_parameters','params':params})

        class FrozenSelection:
            def fit(self, X, y):
                self.best_params_ = params
                self.best_index_ = 0
                self.cv_results_ = {k:[float(selected[k.replace('test_score','valid_score')])] for k in ('mean_train_score','std_train_score','mean_test_score','std_test_score')}
                self.best_estimator_ = clone(kwargs['estimator']).set_params(**params).fit(X,y)
                return self
        return FrozenSelection()

    benchmark.RandomizedSearchCV = make_search
    features = ['brand+name+price+desc'] if a.model == 'wordcooc' else [['brand','name','desc','price']]
    getattr(benchmark, function_name)(str(train),str(valid),str(test),features,benchmark.classifiers,experiment,write_test_set_for_inspection=False)
    (output/'parameter_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
(output/'complete.json').write_text(json.dumps({'seed':a.seed,'tests':[str(p) for p in tests]})+'\n')
