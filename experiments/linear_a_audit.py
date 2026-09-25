"""Repair audit: python -m experiments.linear_a_audit freeze|verify|profiles|rules|trade."""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from experiments import linear_a_probes as old, linear_a_tlhdig as previous
from linear_a import audit, probes, probes_v2 as fixed
from linear_a.controls import rng_for
from linear_a.matching import BigramNull
from linear_a.spelling import parse_syllabic, render

OUT = Path('experiments/linear-a-audit')
ALPHA = .05/7
RULE_RUNS = 9999
TRADE_RUNS = 999
SAMPLES = 20


def freeze():
    files = [*Path('linear_a').glob('*.py'), 'experiments/linear_a_audit.py',
             'experiments/linear_a_probes.py', 'experiments/linear_a_tlhdig.py',
             'experiments/linear_a_context.py', 'experiments/linear-a-probes/lists.json',
             'experiments/linear-a-audit/PROTOCOL.md', 'tests/test_linear_a_v2.py']
    sources = ['artifacts/linear-a-sources/navarre/corpus.json',
               'artifacts/linear-a-sources/kaikki/AncientGreek.jsonl',
               'artifacts/linear-a-sources/tlhdig/forms.json',
               *Path('artifacts/linear-a-sources/damos/items').glob('*.json')]
    audit.freeze(OUT, files, sources)


def unique_nearest(target, references, tag):
    rng = rng_for('repair-profiles', tag)
    targets, refs = [], {k: [] for k in references}
    quotas_record = []
    for _ in range(SAMPLES):
        quotas = fixed.common_quotas([target, *references.values()], 300, rng)
        quotas_record.append(dict(sorted(quotas.items())))
        targets.append(fixed.profile(fixed.matched_unique(target, quotas, rng)))
        for k, pool in references.items():
            refs[k].append(fixed.profile(fixed.matched_unique(pool, quotas, rng)))
    winners, distances = fixed.nearest_profiles(targets, refs)
    return {'nearest': dict(collections.Counter(winners)), 'sample_size': sum(quotas.values()),
            'length_quotas': quotas_record, 'distances': distances,
            'target_profiles': [p.tolist() for p in targets],
            'reference_centres': {k: np.mean(v, axis=0).tolist() for k, v in refs.items()}}


def shuffled(words, rng):
    signs = [s for w in words for s in w]
    order = rng.permutation(len(signs))
    out, start = [], 0
    for w in words:
        out.append(tuple(signs[i] for i in order[start:start+len(w)]))
        start += len(w)
    return out


def profile_checks():
    lists = previous.spelled_lists()
    la = sorted(old.linear_a()['labels'])
    lb = sorted(old.linear_b()['labels'])
    # Recreate the historical method and make its formerly interactive checks executable.
    # These are retrospective diagnostics, not independent confirmatory data.
    rng = rng_for('audit-diagnostics')
    pseudo = BigramNull(la, rng).samples([len(w) for w in la], 1)[0]
    shuffle = shuffled(la, rng)
    small = {**lists, 'Hittite': [lists['Hittite'][i] for i in
                                rng.choice(len(lists['Hittite']), 700, replace=False)]}
    merge = lambda ws: sorted(set(tuple((c, 'u' if v == 'o' else v) for c, v in w) for w in ws))
    merged = {k: merge(v) for k, v in lists.items()}
    legacy = {}
    for name, target, refs, tag in [
        ('linear_b', lb, lists, 'linear-b'), ('linear_a', la, lists, 'linear-a'),
        ('linear_a_shuffled', shuffle, lists, 'audit-shuffle'),
        ('linear_a_bigram', pseudo, lists, 'audit-bigram'),
        ('hittite_700', la, small, 'audit-small'),
        ('merge_o_u', merge(la), merged, 'audit-merged'),
        ('linear_b_merge_o_u', merge(lb), merged, 'audit-merged-b')]:
        counts, size = previous.nearest_counts(target, refs, tag)
        legacy[name] = {'nearest': counts, 'sample_size': size}
    print('historical diagnostics', legacy, flush=True)
    controls = {}
    for k in previous.NAMES.values():
        halves = {h: [w for w in lists[k] if probes.half(w) == h] for h in (0, 1)}
        controls[k] = unique_nearest(halves[1], {**lists, k: halves[0]}, 'self-'+k)
        print('repaired self control', k, controls[k]['nearest'], controls[k]['sample_size'], flush=True)
    control_b = unique_nearest(lb, lists, 'linear-b')
    control_shuffle = unique_nearest(shuffled(lb, rng_for('repair-shuffle-b')), lists, 'shuffle-b')
    # A method used as a word-structure test must not identify shuffled Linear B as Greek.
    positive = control_b['nearest'].get('Greek', 0) >= 18
    negative = control_shuffle['nearest'].get('Greek', 0) <= 1
    eligible = [k for k, r in controls.items() if r['nearest'].get(k, 0) >= 18]
    gate = positive and negative and bool(eligible)
    targets = None
    if gate:
        targets = {k: unique_nearest(ws, lists, k) for k, ws in
                   [('linear_a', la), ('linear_a_shuffled', shuffle), ('linear_a_bigram', pseudo)]}
    result = {'historical_diagnostics': legacy, 'features': probes.FEATURES,
              'controls': controls, 'linear_b': control_b, 'shuffled_linear_b': control_shuffle,
              'gate': {'positive': positive, 'negative': negative, 'eligible': eligible, 'passed': gate},
              'linear_a': targets, 'interpretation': 'No language claim. Target skipped if gate fails.'}
    audit.save(OUT/'profiles.json', result)
    print('profile gate', result['gate'], flush=True)


def rule_checks():
    fixed.require_resolution(RULE_RUNS, ALPHA)
    words = sorted(old.linear_a()['labels'])
    targets = sorted(old.linear_b()['labels'])
    index = fixed.SubstitutionIndex(targets)
    halves = {h: [w for w in words if probes.half(w) == h] for h in (0, 1)}
    discovery = index.count(halves[0])
    null0 = [index.count(c) for c in probes.null_corpora(halves[0], old.rng(6, 0), 100)]
    keys = []
    for k, n in discovery.items():
        arr = np.array([c[k] for c in null0])
        if n >= 3 and (n-arr.mean())/max(arr.std(), 1.) >= 3:
            keys.append(k)
    keys.sort(key=probes.rule_key)
    results = {'discovered': [probes.rule_key(k) for k in keys],
               'status': 'Retrospective correction on already inspected data; no fresh confirmation.'}
    for label, pool, generator in [('confirmation', halves[1], old.rng(6, 1)),
                                   ('pre_named', words, old.rng(6, 'all'))]:
        null = BigramNull(pool, generator)
        values, per_rule = [], {k: [] for k in keys}
        observed = index.count(pool)
        for i in range(RULE_RUNS):
            counts = index.count([null.sample(len(w)) for w in pool])
            values.append(sum(counts[k] for k in keys) if label == 'confirmation' else old._pre_named(counts))
            if label == 'confirmation':
                for k in keys:
                    per_rule[k].append(counts[k])
            if (i+1) % 1000 == 0:
                print(label, i+1, flush=True)
        obs = sum(observed[k] for k in keys) if label == 'confirmation' else old._pre_named(observed)
        results[label] = fixed.monte_carlo(obs, values, ALPHA)
        results[label]['null_counts'] = values
        if label == 'confirmation':
            # Individual rules share the family's allowance; pooled discovery remains primary.
            results['per_rule'] = {probes.rule_key(k): fixed.monte_carlo(observed[k], vs, ALPHA/max(1, len(keys)))
                                   for k, vs in per_rule.items()}
        print(label, {k:v for k,v in results[label].items() if k!='null_counts'}, flush=True)
    audit.save(OUT/'rules.json', results)


def trade_checks():
    fixed.require_resolution(TRADE_RUNS, ALPHA)
    words = sorted(old.linear_b()['labels'])
    size = len(old.linear_a()['labels'])
    stems = {probes.stem(parse_syllabic(item['linear_b']))
             for item in old.LISTS['trade_words']['items'] if item['linear_b']}
    stems.discard(None)
    # Equality here is the frozen two-sign prefix rule, just indexed instead of nested loops.
    score = lambda ws: len(stems & {probes.stem(w) for w in ws})
    results = []
    for draw in range(20):
        # Independent streams prevent a larger null budget from changing later real samples.
        rng = rng_for('repair-trade', draw)
        sample = [words[i] for i in rng.choice(len(words), size, replace=False)]
        null = BigramNull(sample, rng)
        counts = [score([null.sample(len(w)) for w in sample]) for _ in range(TRADE_RUNS)]
        result = fixed.monte_carlo(score(sample), counts, ALPHA)
        result['null_counts'] = counts
        results.append(result)
        print('trade', draw, result['observed'], result['p'], flush=True)
    passed = sum(r['below_alpha'] for r in results)
    audit.save(OUT/'trade.json', {'draws': results, 'passing_draws': passed, 'required': 18,
                                'passed': passed >= 18, 'linear_a_run': False})


def main():
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    if command == 'freeze':
        freeze()
        return
    if command not in ('verify', 'profiles', 'rules', 'trade'):
        raise SystemExit(__doc__)
    audit.verify(OUT)
    if command == 'verify':
        print('Freeze verified')
        return
    filename = {'profiles': 'profiles.json', 'rules': 'rules.json', 'trade': 'trade.json'}[command]
    if (OUT/filename).exists():
        raise SystemExit(f'{filename} exists; refusing to overwrite')
    {'profiles': profile_checks, 'rules': rule_checks, 'trade': trade_checks}[command]()


if __name__ == '__main__':
    main()
