"""Frozen entry-ending experiment: python -m experiments.linear_a_structure freeze|verify|run."""

import collections
import sys
from pathlib import Path

from experiments.linear_a_probes import linear_b
from linear_a import audit, corpus, structure
from linear_a.controls import rng_for

OUT = Path('experiments/linear-a-structure')
NULL_RUNS = 199


def freeze():
    files = [*Path('linear_a').glob('*.py'), 'experiments/linear_a_structure.py',
             'experiments/linear_a_probes.py', 'experiments/linear_a_context.py',
             'experiments/linear-a-probes/lists.json',
             'experiments/linear-a-structure/PROTOCOL.md', 'tests/test_linear_a_structure.py']
    sources = ['artifacts/linear-a-sources/navarre/corpus.json',
               *Path('artifacts/linear-a-sources/damos/items').glob('*.json')]
    audit.freeze(OUT, files, sources)


def size_like(rows, reference, rng):
    counts = collections.Counter(r['label'] for r in reference)
    result = []
    for y, n in sorted(counts.items()):
        pool = [r for r in rows if r['label'] == y]
        if len(pool) < n:
            raise ValueError('known-answer pool too small for target class counts')
        result.extend(pool[i] for i in rng.choice(len(pool), n, replace=False))
    return result


def evaluate(train, test, predictions=False):
    labels = [r['label'] for r in test]
    base = structure.predict(train, test)
    ending = structure.predict(train, test, endings=True)
    a = structure.balanced_accuracy(labels, base)
    b = structure.balanced_accuracy(labels, ending)
    result = {'train_types': len(train), 'test_types': len(test), 'baseline_ba': a,
              'endings_ba': b, 'gain': b-a, 'passed': b >= .60 and b-a >= .05,
              'endings_class_recalls': {str(y): float(np.mean(ending[np.array(labels)==y] == y))
                                       for y in (0,1)}}
    if predictions:
        result['predictions'] = [{**r, 'baseline': int(p), 'endings': int(q)}
                                 for r,p,q in zip(test,base,ending)]
    return result


def null_check(train, test, tag):
    observed = evaluate(train, test)
    values = []
    rng = rng_for('entry-endings-null', tag)
    for _ in range(NULL_RUNS):
        shuffled = structure.permuted_training(train, rng)
        values.append(evaluate(shuffled, test)['gain'])
    # No integer conversion: the statistic here is a difference of balanced accuracies.
    p = (sum(v >= observed['gain'] for v in values)+1)/(len(values)+1)
    return {'observed_gain': observed['gain'], 'null_gains': values, 'p': p,
            'minimum_p': 1/(len(values)+1), 'null_runs': len(values)}


def main():
    command = sys.argv[1] if len(sys.argv)==2 else ''
    if command == 'freeze':
        freeze()
        return
    if command not in ('verify', 'run'):
        raise SystemExit(__doc__)
    audit.verify(OUT)
    if command == 'verify':
        print('Freeze verified')
        return
    if (OUT/'results.json').exists():
        raise SystemExit('results.json exists; refusing to overwrite')
    rows, known = structure.sign_tokens(corpus.load())
    la_train, la_test = structure.partition(rows)
    b_rows = [(doc, tuple(c+v for c,v in w), int(label=='entry'), site)
              for doc,w,label,series,site in linear_b()['rows']]
    lb_train, lb_test = structure.partition(b_rows)
    full = evaluate(lb_train, lb_test)
    print('full Linear B control', full, flush=True)
    controls, negatives = [], []
    for draw in range(20):
        rng = rng_for('entry-endings-control', draw)
        train = size_like(lb_train, la_train, rng)
        test = size_like(lb_test, la_test, rng)
        controls.append(evaluate(train, test))
        negatives.append(evaluate(structure.shuffled_words(train, rng),
                                  structure.shuffled_words(test, rng)))
    positive_n = sum(r['passed'] for r in controls)
    negative_n = sum(r['passed'] for r in negatives)
    gate = positive_n >= 18 and negative_n <= 1
    result = {'question': 'Do final sign sequences predict numeric-entry position for unseen word types?',
              'linear_a_coverage': {'tokens': len(rows), 'documents': len({r[0] for r in rows}),
                  'train_types': len(la_train), 'test_types': len(la_test),
                  'train_classes': dict(collections.Counter(r['label'] for r in la_train)),
                  'test_classes': dict(collections.Counter(r['label'] for r in la_test)),
                  'test_types_with_unknown_sign_readings': sum(any(s not in known for s in r['word']) for r in la_test)},
              'full_linear_b_control': full, 'size_matched_controls': controls,
              'shuffled_controls': negatives,
              'gate': {'positive_passes': positive_n, 'negative_passes': negative_n, 'passed': gate},
              'linear_b_label_shuffle': null_check(lb_train, lb_test, 'linear-b'), 'linear_a': None}
    if gate:
        target = evaluate(la_train, la_test, predictions=True)
        target['label_shuffle'] = null_check(la_train, la_test, 'linear-a')
        rng = rng_for('entry-endings-target-shuffle')
        target['within_word_shuffle'] = evaluate(structure.shuffled_words(la_train,rng),
                                               structure.shuffled_words(la_test,rng))
        target['lead'] = (target['passed'] and target['label_shuffle']['p'] <= .05
                          and not target['within_word_shuffle']['passed'])
        result['linear_a'] = target
    audit.save(OUT/'results.json', result)
    print('gate', result['gate'], 'Linear A scored:', result['linear_a'] is not None, flush=True)


if __name__ == '__main__':
    main()
