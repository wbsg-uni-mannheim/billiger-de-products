"""Build local v1.1 evaluation sets with explicit rule/AI label provenance.

python -m src.processing.finalize_v11 --output-root data/releases/v1.1-local
Original releases and intermediate drafts are never overwritten.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from .prepare_v11_draft import (
    CORE, FAMILIES, LANGUAGES, TEST, TRAIN, VALID, digest, exact_test_evidence,
    inspect_splits, json_hash, load_source, offer_ids, pair_id, pair_key,
    paths_for, product_offers, read_rows, write_csv, write_rows,
)


def product_key(row):
    return tuple(sorted((row['product_id_left'], row['product_id_right'])))


def payload(rows):
    result = {}
    for lang, row in rows.items():
        offers = [{k[:-len(side)-1]: v for k, v in row.items()
                   if k.endswith('_' + side)} for side in ('left', 'right')]
        result[lang] = sorted(offers, key=lambda o: int(o['id']))
    return result


def statistics(rows):
    return dict(pairs=len(rows), offers=len(offer_ids(rows)),
                products=len(product_offers(rows)),
                positives=sum(r['label'] == 1 for r in rows),
                negatives=sum(r['label'] == 0 for r in rows),
                unresolved=sum(r['label'] is None for r in rows),
                hard_negative_flags=sum(bool(r['is_hard_negative']) for r in rows))


def finalize(draft, review, output):
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    parent = json.loads((draft / 'manifest.json').read_text())
    for relative, expected in parent['candidate_sha256'].items():
        assert digest(draft / relative) == expected, relative
    source = Path(parent['source_root'])
    original, registries, hashes, _ = load_source(source)
    assert hashes == parent['source_sha256'], 'Original source changed'
    evidence, evidence_files = exact_test_evidence(original['de'])
    cross_matches = {product_key(r) for p, rows in original['de'].items()
                     if p.parts[0] == TEST for r in rows
                     if r['label'] == 1 and r['product_id_left'] != r['product_id_right']}
    plans = json.loads((review / 'replacement_plan_49.json').read_text())
    assert len(plans) == 49
    replacements = {(p['file'], p['old_pair_id']): p for p in plans}
    assert len(replacements) == len(plans)
    removed_reviews = {p['old_review_id'] for p in plans}
    ai = {x['review_id']: x for x in json.loads((review / 'ai_review.json').read_text())
          if x['case_id'].startswith('E') and x['review_id'] not in removed_reviews}
    assert len(ai) == 15 and all(x['confidence'] == 'high' for x in ai.values())
    decisions, ledger = {}, []
    with (draft / 'review/label_provenance.jsonl').open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['resolved_label'] is not None:
                continue
            rid = item['review_id']
            folders = {Path(o['file']).parts[0] for o in item['original_occurrences']}
            assert len(folders) == 1
            folder = next(iter(folders))
            offers = item['new_offers']['de']
            products = tuple(sorted(o['product_id'] for o in offers))
            entry = {k: item[k] for k in ('review_id', 'pair_id', 'content_sha256')}
            entry['files'] = sorted({o['file'] for o in item['original_occurrences']})
            if rid in removed_reviews:
                origin, label = 'replaced_unresolvable_pair', None
            elif rid in ai:
                decision = ai[rid]
                assert decision['content_sha256'] == item['content_sha256']
                origin, label = 'ai_text_review_high_confidence', decision['ai_label']
                entry.update(reviewer=decision['reviewer'], rationale=decision['rationale'])
            elif folder == VALID:
                origin, label = 'validation_source_product_id_rule', int(products[0] == products[1])
            else:
                assert folder == TEST and products[0] != products[1] and products not in cross_matches, rid
                origin, label = 'test_cross_product_rule_no_known_match', 0
            entry.update(label_source=origin, resolved_label=label)
            ledger.append(entry)
            for file in entry['files']:
                decisions[(file, item['pair_id'])] = entry
    counts = Counter(x['label_source'] for x in ledger)
    assert counts == {'replaced_unresolvable_pair': 49,
                      'ai_text_review_high_confidence': 15,
                      'validation_source_product_id_rule': 12296,
                      'test_cross_product_rule_no_known_match': 6147}, counts
    loaded = {lang: {} for lang in LANGUAGES}
    stats, applied = [], set()
    for path in original['de']:
        rows = {lang: read_rows(draft / 'candidate_sets' / f'solute_{lang}' / path)
                for lang in LANGUAGES}
        family = next(f for f in FAMILIES if f in path.name)
        seen = set(product_offers(original['de'][paths_for(family, TRAIN)]))
        result = {lang: [] for lang in LANGUAGES}
        for index, row in enumerate(rows['de']):
            identity = (str(path), pair_id(pair_key(row)))
            paired = {lang: rows[lang][index].copy() for lang in LANGUAGES}
            decision = decisions.get(identity)
            if identity in replacements:
                plan = replacements[identity]
                assert row['label'] is None
                assert json_hash(payload(paired)) == plan['old_content_sha256']
                new = plan['replacement']
                key = pair_key(new)
                assert evidence[key] == {new['label']}
                assert set(plan['existing_test_files']) == evidence_files[key]
                assert set(key) <= offer_ids(rows['de'])
                assert pair_id(key) not in {pair_id(pair_key(r)) for r in rows['de']}
                assert all(row['product_id_' + s] in seen and new['product_id_' + s] in seen
                           for s in ('left', 'right'))
                paired = {}
                for lang in LANGUAGES:
                    paired[lang] = {k: new[k] for k in ('pair_id', 'label', 'is_hard_negative')}
                    for side in ('left', 'right'):
                        offer = registries[lang][TEST][new['id_' + side]]
                        paired[lang].update({k + '_' + side: v for k, v in offer.items()})
                assert json_hash(payload(paired)) == plan['replacement_content_sha256']
                applied.add(identity)
            elif row['label'] is None:
                assert decision and decision['resolved_label'] in (0, 1), identity
                assert json_hash(payload(paired)) == decision['content_sha256']
                for lang in LANGUAGES:
                    paired[lang]['label'] = decision['resolved_label']
            for lang in LANGUAGES:
                result[lang].append(paired[lang])
        assert offer_ids(result['de']) == offer_ids(rows['de']), path
        assert set(product_offers(result['de'])) == set(product_offers(rows['de'])), path
        if path.parts[0] == TEST and 'rnd050un' in path.name:
            def unseen_only(rr):
                return [r for r in rr if all(r['product_id_' + s] not in seen
                                            for s in ('left', 'right'))]
            assert unseen_only(result['de']) == unseen_only(rows['de'])
        for lang in LANGUAGES:
            rr = result[lang]
            assert len(rr) == len(rows[lang]) == len({pair_key(r) for r in rr})
            assert all(r['label'] in (0, 1) and r['id_left'] != r['id_right'] for r in rr)
            loaded[lang][path] = rr
        assert all(all(a[k] == b[k] for k in CORE)
                   for a, b in zip(result['de'], result['en']))
        stats.append({'file': str(path), **{'original_' + k: v for k, v in statistics(original['de'][path]).items()},
                      **{'draft_' + k: v for k, v in statistics(rows['de']).items()},
                      **{'final_' + k: v for k, v in statistics(result['de']).items()}})
    assert applied == set(replacements)
    checks = {}
    for lang in LANGUAGES:
        overlaps, conflicts = inspect_splits(loaded[lang])
        assert not conflicts, conflicts
        assert all(not x['shared_offers'] and not x['shared_pairs'] for x in overlaps)
        checks[lang] = {'split_checks': overlaps, 'global_label_conflicts': conflicts}
    output.mkdir(parents=True)
    audit = output / 'review'
    audit.mkdir()
    output_hashes, frozen_count = {}, 0
    for lang, files in loaded.items():
        for path, rows in files.items():
            dest = output / 'candidate_sets' / f'solute_{lang}' / path
            frozen = path.parts[0] == TRAIN or (path.parts[0] == VALID and 'rnd000un' in path.name) or (path.parts[0] == TEST and 'rnd100un' in path.name)
            if frozen:
                assert rows == original[lang][path]
                frozen_count += 1
            if rows == read_rows(draft / 'candidate_sets' / f'solute_{lang}' / path):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(draft / 'candidate_sets' / f'solute_{lang}' / path, dest)
            else:
                write_rows(dest, rows)
            if frozen:
                assert digest(dest) == hashes[str(Path(f'solute_{lang}') / path)]
            output_hashes[str(dest.relative_to(output))] = digest(dest)
    assert frozen_count == 42
    write_csv(audit / 'file_statistics.csv', stats, list(stats[0]))
    with (audit / 'label_decisions.jsonl').open('w') as stream:
        for entry in ledger:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    shutil.copy2(review / 'replacement_plan_49.json', audit / 'replacement_plan_49.json')
    replacement_table = [dict(case_id=p['case_id'], file=p['file'], old_pair_id=p['old_pair_id'],
                              new_pair_id=p['replacement']['pair_id'],
                              label=p['replacement']['label'],
                              de_left=p['replacement']['name_left'],
                              de_right=p['replacement']['name_right'],
                              rationale=p['rationale'], reviewer=p['reviewer']) for p in plans]
    write_csv(audit / 'replacement_pairs.csv', replacement_table, list(replacement_table[0]))
    shutil.copy2(Path(__file__), output / 'finalizer_snapshot.py')
    (audit / 'structural_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    manifest = dict(parent)
    manifest.update(version='1.1-local', status='LOCAL_EVALUATION_READY_RULE_AND_AI_LABELS',
                    ready_for_inference=True, blockers={k: 0 for k in parent['blockers']},
                    candidate_sha256=output_hashes, parent_draft=str(draft.resolve()),
                    parent_manifest_sha256=digest(draft / 'manifest.json'),
                    finalizer_sha256=digest(Path(__file__)), label_decision_counts=dict(counts),
                    replacement_pairs_per_language=49, frozen_files_verified=42,
                    label_rule='Exact original pair labels for replacements; 12296 unique validation labels by product-ID equality; 6147 unique test labels by different product IDs with no known cross-product match; 15 high-confidence AI text reviews. No new human adjudication.',
                    hardness_rule='Replacement rows reuse original source-pair sampling flags. Other remapped rows retain inherited sampling flags. Actual text difficulty has not been measured.',
                    selection_rule='Purposive clarity selection from unused audited pairs of already present Seen offers; no model predictions used. Pair/product-category weighting and class balance may change.',
                    audit_sha256={str(p.relative_to(output)): digest(p) for p in audit.iterdir()},
                    input_review_sha256={name: digest(review / name) for name in ('ai_review.json', 'replacement_plan_49.json')})
    # Parent aggregate label-change counts describe an earlier stage, not this build.
    manifest.pop('known_label_changes_per_language', None)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(output), 'decisions': dict(counts),
                      'half_seen': [s for s in stats if TEST in s['file'] and 'rnd050un' in s['file']]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, default=Path('data/releases/v1.1-drop102-draft'))
    parser.add_argument('--review', type=Path, default=Path('reports/v11_label_spotcheck_2026-09-06'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.draft, args.review, args.output_root)


if __name__ == '__main__':
    main()
