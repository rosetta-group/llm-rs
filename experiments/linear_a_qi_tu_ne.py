"""Bounded source collation of qi-tu-ne: freeze|verify|run, no semantic scoring."""
import json
import sys
from pathlib import Path
from linear_a import audit, corpus, person_slots

OUT = Path('experiments/linear-a-qi-tu-ne')


def main(command):
    inputs = ('PROTOCOL.md', 'inventory.json', 'cases.json', 'rivals.json', 'sources.json')
    if command == 'freeze':
        sources = json.loads((OUT / 'sources.json').read_text())
        audit.freeze(OUT, ['experiments/linear_a_qi_tu_ne.py', 'linear_a/person_slots.py',
                          'linear_a/audit.py', 'linear_a/corpus.py', 'linear_a/spelling.py',
                          *[OUT / name for name in inputs]],
                     [corpus.CORPUS, *[s['local_path'] for s in sources['files']]])
    elif command in ('verify', 'run'):
        audit.verify(OUT)
        data = corpus.load()
        inventory = json.loads((OUT / 'inventory.json').read_text())
        cases = json.loads((OUT / 'cases.json').read_text())['occurrences']
        person_slots.validate(inventory, data)
        literal = person_slots.literal_occurrences(data, ['qi-tu-ne'])['qi-tu-ne']
        expected = [(r['document'], r['word_index']) for r in cases]
        actual = [(r['document'], r['word_index']) for r in literal['hits']]
        if len(set(expected)) != len(expected) or sorted(expected) != sorted(actual):
            raise ValueError('Reviewed cases do not cover the literal occurrence list exactly')
        for case in cases:
            record = data[case['document']]
            if case['raw'] != record['unicode_text'].splitlines()[case['logical_row'] - 1]:
                raise ValueError('Changed original comparison row')
            signs = [s['type'] for s in record['signs'] if s['n'] in case['sign_n']]
            if signs != ['AB21f', 'AB69', 'AB24']:
                raise ValueError('Different sign sequence in comparison')
            if case['object'] != (record.get('parent_object') or case['document']):
                raise ValueError('Different physical-object key')
        if command == 'run':
            result = person_slots.summary(inventory)
            result['comparison'] = {
                'word': 'qi-tu-ne', 'signs': ['AB21f', 'AB69', 'AB24'],
                'reviewed_occurrences': len(cases),
                'reviewed_objects': len({r['object'] for r in cases}),
                'supplied_roles': {role: sum(r['role'] == role for r in cases)
                                   for role in sorted({r['role'] for r in cases})},
                'note': 'Role labels are supplied source annotations, not model predictions.',
            }
            audit.save(OUT / 'results.json', result)
            with (OUT / 'INVENTORY.md').open('x') as handle:
                # The generic table helper carries its original title; specialize only display text.
                text = person_slots.markdown_inventory(inventory)
                handle.write(text.replace('# HT 85/117 source inventory', '# HT7 source inventory', 1))
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main(sys.argv[1])
