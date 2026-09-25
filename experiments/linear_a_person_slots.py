"""HT85/117 descriptive source audit: freeze|verify|run; exclusive outputs."""
import json
import sys
from pathlib import Path
from linear_a import audit, corpus, person_slots

OUT = Path('experiments/linear-a-person-slots')


def main(command):
    inputs = ('PROTOCOL.md', 'inventory.json', 'queries.json', 'source-review.json',
              'candidate-review.json', 'sources.json')
    if command == 'freeze':
        sources = json.loads((OUT / 'sources.json').read_text())
        audit.freeze(OUT, ['linear_a/person_slots.py', 'linear_a/audit.py',
                          'linear_a/corpus.py', 'linear_a/spelling.py',
                          'experiments/linear_a_person_slots.py',
                          'tests/test_linear_a_person_slots.py',
                          *[OUT / name for name in inputs]],
                     [corpus.CORPUS, *[s['local_path'] for s in sources['files']]])
    elif command in ('verify', 'run'):
        audit.verify(OUT)
        inventory = json.loads((OUT / 'inventory.json').read_text())
        data = corpus.load()
        person_slots.validate(inventory, data)
        if command == 'run':
            queries = json.loads((OUT / 'queries.json').read_text())['literal_words']
            audit.save(OUT / 'results.json', person_slots.summary(inventory))
            audit.save(OUT / 'literal-occurrences.json', person_slots.literal_occurrences(data, queries))
            with (OUT / 'INVENTORY.md').open('x') as handle:
                handle.write(person_slots.markdown_inventory(inventory))
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main(sys.argv[1])
