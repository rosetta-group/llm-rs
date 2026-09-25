"""Correspondence source audit: python -m experiments.linear_a_correspondence prepare|freeze|verify|run."""

import collections
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

from experiments.linear_a_probes import linear_a, linear_b
from linear_a import audit, correspondence as c, corpus
from linear_a.probes_v2 import monte_carlo
from linear_a.spelling import parse_syllabic, render
from linear_a.contexts import _clean_b

OUT = Path('experiments/linear-a-correspondence')
DAMOS = Path('artifacts/linear-a-sources/damos/items')
ALPHA = .05 / 7
REPEATS = 9999
SEEDS = {'strict': {'length': 24092401, 'length_onset': 24092402},
         'historical': {'length': 24092403, 'length_onset': 24092404}}


def prepare():
    """Source extraction only; no real-data permutations until committed freeze."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = corpus.load()
    a_rows = list(c.a_tokens(data))
    b_rows, items = [], {}
    for path in sorted(DAMOS.glob('*.json')):
        item = json.loads(path.read_text())['item']
        if not item:
            continue
        items[path.stem] = item
        for row in c.b_tokens(item):
            b_rows.append({**row, 'id': path.stem, 'document': item['heading_short'],
                           'url': item.get('permalink1')})
    strict_a = sorted({r['word'] for r in a_rows})
    strict_b = sorted({r['word'] for r in b_rows})
    historical_a = sorted({tuple(render(w).split('-')) for w in linear_a()['labels'] if len(w)>=3})
    historical_b = sorted({tuple(render(w).split('-')) for w in linear_b()['labels'] if len(w)>=3})
    audit.save(OUT/'inputs.json', {'strict': {'a': strict_a, 'b': strict_b},
                                 'historical': {'a': historical_a, 'b': historical_b}})
    audit.save(OUT/'accepted-attestations.json', {'a': a_rows, 'b': b_rows})
    original = json.loads(Path('experiments/linear-a-probes/results-6.json').read_text())['pre_named_pairs']
    evidence = []
    for a, b in original:
        na, nb = parse_syllabic(a), parse_syllabic(b)
        a_docs = sorted(linear_a()['docs'][na])
        a_evidence = []
        for doc in a_docs:
            record = data[doc]
            a_evidence.append({'document': doc,
                'url': 'https://sigla.phis.me/document/' + quote(doc, safe='') + '/index-word.html',
                'words': record.get('words'), 'signs': record.get('signs'),
                'unicode_text': record.get('unicode_text'), 'conflicts': record.get('conflicts', []),
                'gorila_ref': record.get('gorila_ref'), 'parent_object': record.get('parent_object'),
                'accepted': [r for r in a_rows if r['document'] == doc
                             and parse_syllabic('-'.join(r['word'])) == na]})
        b_evidence = []
        for id_, item in items.items():
            for line in (item.get('content') or '').splitlines():
                # Historical extraction is reproduced solely to locate source evidence.
                for raw in re.split(r'[\s,/]+', line):
                    if parse_syllabic(_clean_b(raw)) == nb:
                        b_evidence.append({'id': id_, 'document': item['heading_short'],
                            'url': item.get('permalink1'), 'raw': raw, 'line': line,
                            'notes': item.get('notes'),
                            'accepted': any(r['id'] == id_ and r['raw'] == raw and r['line'] == line
                                            for r in b_rows)})
        corrected = [[list(x), list(y)] for x, y in c.pairs(strict_a, strict_b)
                     if parse_syllabic('-'.join(x)) == na and parse_syllabic('-'.join(y)) == nb]
        evidence.append({'historical_pair': [a, b], 'strict_pairs': corrected,
                         'linear_a': a_evidence, 'linear_b': b_evidence})
    audit.save(OUT/'pair-evidence.json', evidence)
    print('Types:', len(strict_a), len(strict_b), 'strict pairs:', c.pairs(strict_a, strict_b))


def freeze():
    files = [*Path('linear_a').glob('*.py'), 'experiments/linear_a_probes.py',
        'experiments/linear_a_context.py', 'experiments/linear_a_correspondence.py',
        'experiments/linear-a-probes/lists.json', 'experiments/linear-a-probes/results-6.json',
        'tests/test_linear_a_correspondence.py',
        *[OUT/n for n in ('PROTOCOL.md', 'PAIRS.md', 'inputs.json', 'accepted-attestations.json',
                          'pair-evidence.json')]]
    audit.freeze(OUT, files, ['artifacts/linear-a-sources/navarre/corpus.json', *DAMOS.glob('*.json')])


def run():
    audit.verify(OUT)
    if (OUT/'results.json').exists():
        raise SystemExit('results.json exists; refusing to overwrite')
    result = {'alpha': ALPHA, 'repeats': REPEATS, 'seeds': SEEDS,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'python': platform.python_version(), 'numpy': np.__version__, 'datasets': {}}
    for name, dataset in json.loads((OUT/'inputs.json').read_text()).items():
        a, b = [[tuple(w) for w in dataset[k]] for k in ('a', 'b')]
        entry = {'a_types': len(a), 'b_types': len(b), 'pairs': c.pairs(a, b),
                 'observed_stems': c.score(a, b), 'nulls': {}}
        for mode in ('length', 'length_onset'):
            values, diagnostics = c.null_scores(a, b, mode, REPEATS, SEEDS[name][mode])
            summary = monte_carlo(entry['observed_stems'], values, ALPHA)
            entry['nulls'][mode] = {**summary, 'diagnostics': diagnostics,
                'distribution': dict(sorted(collections.Counter(values).items())), 'scores': values}
            print(name, mode, summary, flush=True)
        entry['passes_both'] = all(n['p'] < ALPHA and n['exceedance_probability_wilson95'][1] < ALPHA
                                   for n in entry['nulls'].values())
        result['datasets'][name] = entry
    result['decision'] = ('retain for independent evidence' if result['datasets']['strict']['passes_both']
                          else 'archive statistical adaptation lead')
    audit.save(OUT/'results.json', result)
    print(result['decision'])


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    if command == 'verify':
        audit.verify(OUT)
        print('Freeze verified')
    elif command in ('prepare', 'freeze', 'run'):
        globals()[command]()
    else:
        raise SystemExit(__doc__)
