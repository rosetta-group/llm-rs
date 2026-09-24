"""freeze | verify | run: a bounded Theban source audit, not a classifier."""
import json
import sys
from pathlib import Path

from linear_a import audit, theban_65

OUT = Path('experiments/linear-b-theban-65')
CORPUS = Path('artifacts/linear-a-sources/damos/items')
INPUTS = ('PROTOCOL.md', 'cases.json', 'quantities.json', 'sources.json')


def validate():
    sources = json.loads((OUT / 'sources.json').read_text())
    cases = json.loads((OUT / 'cases.json').read_text())['cases']
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)) or set(ids) != set(sources['corpus']['selected_ids']):
        raise ValueError('Duplicate or changed selected objects')
    for case in cases:
        raw = json.loads((CORPUS / f"{case['id']}.json").read_text())
        if (raw['item']['content'] != case['raw_content'] or
                raw['item']['heading_short'] != case['document'] or
                raw['meta']['Joins'] != case['joins']):
            raise ValueError('Source transcription or joined-object identity changed')
    for source in sources['files']:
        if audit.digest(source['local_path']) != source['sha256']:
            raise ValueError('Source hash mismatch')
    quantities = json.loads((OUT / 'quantities.json').read_text())
    for key in ('accounts', 'conversions'):
        if any(row['object'] not in ids for row in quantities[key]):
            raise ValueError('Unpinned quantity object')
    return quantities, cases, sources


def main(command):
    if command == 'freeze':
        _, cases, sources = validate()
        audit.freeze(OUT, ['linear_a/theban_65.py', 'linear_a/audit.py',
                          'experiments/linear_b_theban_65.py',
                          'tests/test_linear_a_theban_65.py',
                          *[OUT / name for name in INPUTS]],
                     [*[CORPUS / f"{c['id']}.json" for c in cases],
                      *[s['local_path'] for s in sources['files']]])
    elif command in ('verify', 'run'):
        audit.verify(OUT)
        quantities, cases, _ = validate()
        if command == 'run':
            audit.save(OUT / 'results.json', theban_65.run(quantities, cases))
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main(sys.argv[1])
