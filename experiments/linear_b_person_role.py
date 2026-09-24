"""Source-backed name/designation challenge: freeze | verify | run."""
import json
import sys
from pathlib import Path
from linear_a import audit, person_role_control as control

OUT = Path('experiments/linear-b-person-role')
INPUTS = ('PROTOCOL.md', 'cases.json', 'public.json', 'labels.json', 'sources.json')


def validate():
    cases = json.loads((OUT / 'cases.json').read_text())['cases']
    sources = json.loads((OUT / 'sources.json').read_text())
    for source in sources['files']:
        if audit.digest(source['local_path']) != source['sha256']:
            raise ValueError('Source digest mismatch')
    objects = {obj: json.loads(Path(path).read_text())['item']
               for obj, path in sources['objects'].items()}
    public, labels = [], {}
    for case in cases:
        source = objects[case['object']]
        if source['heading_short'] != case['document']:
            raise ValueError('Source object mismatch')
        public.append(control.public_case(case, source['content']))
        labels[case['id']] = case['label']
    if public != json.loads((OUT / 'public.json').read_text()):
        raise ValueError('Public features differ from annotated source')
    if labels != json.loads((OUT / 'labels.json').read_text()):
        raise ValueError('Label key differs from source annotations')
    if len(labels) != len(cases):
        raise ValueError('Duplicate case IDs')
    return public, labels


def main(command):
    if command == 'freeze':
        validate()
        sources = json.loads((OUT / 'sources.json').read_text())
        audit.freeze(OUT, ['experiments/linear_b_person_role.py', 'linear_a/person_role_control.py',
                          'linear_a/audit.py', 'tests/test_linear_a_person_role.py',
                          *[OUT / name for name in INPUTS]],
                     [s['local_path'] for s in sources['files']])
    elif command in ('verify', 'run'):
        audit.verify(OUT)
        rows, labels = validate()
        if command == 'run':
            if (OUT / 'results.json').exists():
                raise FileExistsError(OUT / 'results.json')
            audit.save(OUT / 'results.json', control.run(rows, labels))
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main(sys.argv[1])
