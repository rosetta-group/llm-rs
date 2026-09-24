"""Accounting pilot: python -m experiments.linear_a_ledger prepare|freeze|verify|run."""

import collections
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from linear_a import audit, contexts, corpus, ledger

OUT = Path('experiments/linear-a-ledger')
DAMOS = Path('artifacts/linear-a-sources/damos/items')
REPEATS = 999
ALPHA = .05


def make_record(doc, group, lines, vocabulary):
    for n, row in enumerate(lines, 1):
        row['line_number'] = n
        if row['kind'] == 'row':
            words = []
            for word in row['words']:
                key = ledger.opaque(word, 'W')
                vocabulary[key] = word
                words.append(key)
            row['words'] = words
            row['quantity'] = {ledger.opaque(k, 'C'): v for k, v in row['quantity'].items()}
    return {'document': doc, 'group': group, 'lines': lines,
            'fold': int(hashlib.sha256(('ledger-v1:' + group).encode()).hexdigest(), 16) % 5}


def coverage(records):
    rows = ledger.examples(records)
    return {'records': len(records), 'eligible_objects': len({r['group'] for r in rows}),
            'eligible_rows': len(rows), 'numeric_lines': sum(r['kind']=='row' for d in records for r in d['lines']),
            'barriers': dict(collections.Counter(r['reason'] for d in records for r in d['lines'] if r['kind']=='barrier')),
            'fold_objects': {str(i): len({r['group'] for r in rows if r['fold']==i}) for i in range(5)}}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    vocab_b, vocab_a, b, dev = {}, {}, [], []
    # Exact duplicate contents are co-grouped with the first object, preventing copy leakage.
    groups_by_text = {}
    for path in sorted(DAMOS.glob('*.json')):
        item = json.loads(path.read_text())['item']
        if not item:
            continue
        text = item.get('content') or ''
        site = item.get('ishort') or '?'
        group = site + ':' + str(item.get('tablenumber') or path.stem)
        signature = (site, ' '.join(text.split()))
        group = groups_by_text.setdefault(signature, group)
        record = make_record(path.stem, group, [ledger.b_line(s) for s in text.splitlines()], vocab_b)
        (b if site == 'KN' else dev).append(record)
    data = corpus.load()
    table = contexts.sign_table(data)
    a, rejected = [], collections.Counter()
    for doc, r in sorted(data.items()):
        if r.get('type') != 'tablet':
            rejected['not_tablet'] += 1
            continue
        if not r.get('signs'):
            rejected['no_sign_layer'] += 1
            continue
        if r.get('conflicts') or any(s.get('certain') is not True for s in r['signs']):
            rejected['source_conflict_or_uncertain_sign'] += 1
            continue
        a.append(make_record(doc, r.get('parent_object') or doc,
                            [ledger.a_line(s, table) for s in (r.get('unicode_text') or '').splitlines()], vocab_a))
    inputs = {'linear_b_control': b, 'linear_b_development': dev, 'linear_a': a}
    audit.save(OUT/'inputs.json', inputs)
    audit.save(OUT/'evaluator-vocabulary.json', {'linear_b': vocab_b, 'linear_a': vocab_a,
        'known_total_words': ['to-so', 'to-sa'],
        'note': 'Only evaluator reads these meanings, after opaque solver outputs are made.'})
    counts = {k: coverage(v) for k, v in inputs.items()}
    counts['linear_a_rejected_records'] = dict(rejected)
    audit.save(OUT/'coverage.json', counts)
    audit.save(OUT/'control-source-review.json', [
        {'id': r['document'], 'group': r['group'],
         'source': json.loads((DAMOS/(r['document']+'.json')).read_text())['item'],
         'parsed_lines': r['lines']}
        for r in b if list(ledger.blocks(r['lines']))])
    print(json.dumps(counts, indent=2))


def freeze():
    files = [*Path('linear_a').glob('*.py'), 'experiments/linear_a_ledger.py',
             'tests/test_linear_a_ledger.py',
             *[OUT/p for p in ('PROTOCOL.md', 'inputs.json', 'evaluator-vocabulary.json', 'coverage.json',
                               'control-source-review.json')]]
    audit.freeze(OUT, files, ['artifacts/linear-a-sources/navarre/corpus.json', *DAMOS.glob('*.json')])


def evaluate(outputs, totals):
    predicted = [r for r in outputs if r['prediction'] is not None]
    positive = [r for r in outputs if set(r['words']) & totals]
    correct = [r for r in predicted if r['prediction']==r['truth']]
    total_correct = [r for r in positive if r['prediction']==r['truth']]
    return {'eligible_rows': len(outputs), 'predictions': len(predicted), 'exact_predictions': len(correct),
        'prediction_precision': len(correct)/len(predicted) if predicted else 0,
        'known_total_rows': len(positive), 'known_total_objects': len({r['group'] for r in positive}),
        'known_total_exact': len(total_correct),
        'known_total_recall': len(total_correct)/len(positive) if positive else 0,
        'known_total_baseline_sum_before': sum(r['candidates'].get('sum_before') == r['truth'] for r in positive),
        'known_total_baseline_sum_after': sum(r['candidates'].get('sum_after') == r['truth'] for r in positive),
        'operator_words': sorted({w for r in predicted for w, _ in r['rules']}),
        'predicted_objects': len({r['group'] for r in predicted})}


def numeric_stat(outputs):
    # One vote per whole physical object, not per face, row, word or commodity.
    groups = collections.defaultdict(list)
    for r in outputs:
        if r['prediction'] is not None:
            groups[r['group']].append(r['prediction']==r['truth'])
    return sum(all(v) for v in groups.values())


def preflight(data):
    b = coverage(data['linear_b_control'])
    a = coverage(data['linear_a'])
    reasons = []
    if b['eligible_objects'] < 10:
        reasons.append('Fewer than 10 eligible control objects; even total-positive coverage cannot reach 10.')
    if a['eligible_objects'] < 10:
        reasons.append('Fewer than 10 eligible target objects.')
    if b['eligible_objects'] < a['eligible_objects']:
        reasons.append('Cannot sample the target number of control objects without replacement.')
    return {'passed': not reasons, 'reasons': reasons,
            'control_eligible_objects': b['eligible_objects'], 'target_eligible_objects': a['eligible_objects']}


def run():
    audit.verify(OUT)
    if (OUT/'results.json').exists():
        raise SystemExit('results.json exists; refusing to overwrite')
    data = json.loads((OUT/'inputs.json').read_text())
    check = preflight(data)
    if not check['passed']:
        result = {'status': 'not_evaluable', 'preflight': check,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'control': None, 'negative_control': None, 'linear_a': None,
            'note': 'No fitting, test predictions or null permutations were run. This is a coverage failure, not evidence against arithmetic structure.'}
        audit.save(OUT/'results.json', result)
        print(json.dumps(result, indent=2))
        return
    # Fit/predict without importing or consulting the evaluator vocabulary.
    rows = ledger.examples(data['linear_b_control'])
    models, outputs = ledger.cross_validate(rows)
    audit.save(OUT/'control-predictions.json', {'models': models, 'predictions': outputs})
    vocabulary = json.loads((OUT/'evaluator-vocabulary.json').read_text())
    totals = {w for w, reading in vocabulary['linear_b'].items() if reading in vocabulary['known_total_words']}
    metrics = evaluate(outputs, totals)
    chosen = set(metrics['operator_words'])
    semantic_precision = len(chosen & totals)/len(chosen) if chosen else 0
    coverage_ok = metrics['known_total_objects'] >= 10
    gate = (coverage_ok and metrics['known_total_recall'] >= .8
            and metrics['prediction_precision'] >= .9 and semantic_precision >= .9)
    result = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'control': metrics, 'semantic_precision': semantic_precision,
              'operator_readings': {w: vocabulary['linear_b'][w] for w in chosen},
              'coverage_gate': coverage_ok, 'performance_gate': gate, 'negative_control': None,
              'linear_a': None}
    print('Control:', json.dumps(result, indent=2), flush=True)
    # Always run the adversarial check, including refitting the full search in every draw.
    rng = np.random.default_rng(24092411)
    nulls, false_gates = [], 0
    for n in range(REPEATS):
        _, shuffled = ledger.cross_validate(ledger.examples(ledger.shuffled_records(data['linear_b_control'], rng)))
        nulls.append(numeric_stat(shuffled))
        m = evaluate(shuffled, totals)
        chosen_null = set(m['operator_words'])
        purity = len(chosen_null & totals)/len(chosen_null) if chosen_null else 0
        false_gates += bool(m['known_total_objects'] >= 10 and m['known_total_recall'] >= .8
                            and m['prediction_precision'] >= .9 and purity >= .9)
        if (n+1) % 200 == 0:
            print('Negative control', n+1, '/', REPEATS, flush=True)
    observed = numeric_stat(outputs)
    p = (1+sum(v>=observed for v in nulls))/(REPEATS+1)
    negative_pass = p < ALPHA and false_gates/REPEATS <= .05
    result['negative_control'] = {'observed_exact_objects': observed, 'null_scores': nulls,
        'p': p, 'null_mean': float(np.mean(nulls)), 'false_gate_passes': false_gates,
        'draws': REPEATS, 'passed': negative_pass}
    result['gate_passed'] = gate and negative_pass
    if result['gate_passed']:
        # Size matching uses whole eligible objects, preserving all their rows and faces.
        a_records = data['linear_a']
        a_rows = ledger.examples(a_records)
        n_target = len({r['group'] for r in a_rows})
        b_groups = sorted({r['group'] for r in rows})
        draws = []
        if n_target <= len(b_groups) and n_target >= 10:
            for draw in range(20):
                sample = set(np.random.default_rng(24092500+draw).choice(b_groups, n_target, replace=False))
                _, out = ledger.cross_validate([r for r in rows if r['group'] in sample])
                m = evaluate(out, totals)
                selected = set(m['operator_words'])
                purity = len(selected & totals)/len(selected) if selected else 0
                draws.append({'metrics': m, 'passed': m['known_total_objects'] >= 3
                    and m['known_total_recall'] >= .8 and m['prediction_precision'] >= .9 and purity >= .9})
        result['size_matched_controls'] = draws
        result['size_gate_passed'] = len(draws)==20 and sum(d['passed'] for d in draws)>=18
        if result['size_gate_passed']:
            a_models, a_outputs = ledger.cross_validate(a_rows)
            a_nulls = []
            rng = np.random.default_rng(24092412)
            for _ in range(REPEATS):
                _, out = ledger.cross_validate(ledger.examples(ledger.shuffled_records(a_records, rng)))
                a_nulls.append(numeric_stat(out))
            a_observed = numeric_stat(a_outputs)
            result['linear_a'] = {'models': a_models, 'predictions': a_outputs,
                'readings': {w: vocabulary['linear_a'][w] for model in a_models.values() for w in model},
                'exact_objects': a_observed, 'null_scores': a_nulls,
                'p': (1+sum(v>=a_observed for v in a_nulls))/(REPEATS+1)}
    audit.save(OUT/'results.json', result)
    print('Control gate:', result['gate_passed'], 'Linear A scored:', result['linear_a'] is not None)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv)==2 else ''
    if command == 'verify':
        audit.verify(OUT)
        print('Freeze verified')
    elif command in ('prepare', 'freeze', 'run'):
        globals()[command]()
    else:
        raise SystemExit(__doc__)
