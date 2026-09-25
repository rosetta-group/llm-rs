"""freeze | verify | run: source-qualified relational-fragment control."""
import json
import sys
from pathlib import Path
from linear_a import audit, relation_control as control

OUT = Path('experiments/linear-b-relations')
CORPUS = Path('artifacts/linear-a-sources/damos/items')
INPUTS = ('PROTOCOL.md', 'cases.json', 'public.json', 'labels.json', 'opaque-key.json', 'sources.json')


def load_records():
    return {p.stem: json.loads(p.read_text())['item'] for p in sorted(CORPUS.glob('*.json'))}


def validate():
    sources = json.loads((OUT / 'sources.json').read_text())
    for source in sources['files']:
        if audit.digest(source['local_path']) != source['sha256']:
            raise ValueError('Source hash mismatch')
    records = load_records()
    cases = json.loads((OUT / 'cases.json').read_text())['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case ID')
    for case in cases:
        item = records[case['object']]
        if item['heading_short'] != case['document']:
            raise ValueError('Object identity changed')
        for line in case.get('raw_lines', []):
            if item['content'].splitlines()[line['line_index']-1] != line['raw']:
                raise ValueError('Review text changed')
        if case['status'] == 'unresolved' and case['label'] is not None:
            raise ValueError('Unresolved case has a gold label')
    rows, vocabulary = control.public_view(cases, records)
    labels = {c['id']: c['label'] for c in cases if c['status'] == 'scored'}
    for filename, value in [('public.json', rows), ('opaque-key.json', vocabulary), ('labels.json', labels)]:
        if value != json.loads((OUT / filename).read_text()):
            raise ValueError(f'Derived input mismatch: {filename}')
    return rows, labels, cases, records


def main(command):
    if command == 'freeze':
        validate()
        sources = json.loads((OUT / 'sources.json').read_text())
        audit.freeze(OUT, ['linear_a/relation_control.py', 'linear_a/audit.py',
                          'experiments/linear_b_relations.py', 'tests/test_linear_a_relations.py',
                          *[OUT / name for name in INPUTS]],
                     [*CORPUS.glob('*.json'), *[s['local_path'] for s in sources['files']]])
    elif command in ('verify', 'run'):
        frozen = audit.verify(OUT)
        # Refuse unpinned added corpus records as well as modified/missing records.
        expected = {p for p in frozen['sources'] if str(p).startswith(str(CORPUS)+'/')}
        if expected != {str(p) for p in CORPUS.glob('*.json')}:
            raise ValueError('Corpus membership changed')
        rows, labels, cases, records = validate()
        if command == 'run':
            outputs = ('results.json', 'marker-inventory.json')
            if any((OUT / name).exists() for name in outputs):
                raise FileExistsError('Output exists; use the archived record')
            audit.save(OUT / 'results.json', control.run(rows, labels, cases))
            audit.save(OUT / 'marker-inventory.json', control.inventory(records))
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main(sys.argv[1])
