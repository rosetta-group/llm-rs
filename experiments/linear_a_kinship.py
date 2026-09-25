"""Descriptive feasibility inventory: freeze|verify|run. No semantic score."""
import json
import sys
from pathlib import Path
from linear_a import audit, corpus, kinship_inventory
OUT=Path('experiments/linear-a-kinship')
DAMOS=Path('artifacts/linear-a-sources/damos/items')

def main(command):
    if command=='freeze':
        audit.freeze(OUT, ['linear_a/kinship_inventory.py','linear_a/contexts.py','linear_a/arithmetic.py',
            'linear_a/corpus.py','linear_a/spelling.py','linear_a/audit.py','experiments/linear_a_kinship.py',
            'tests/test_linear_a_kinship.py',*[OUT/n for n in ('PROTOCOL.md','controls.json','sources.json')]],
            [corpus.CORPUS,*DAMOS.glob('*.json'), 'artifacts/linear-a-sources/account-benchmark/killen-tu.pdf'])
    elif command in ('run','verify'):
        audit.verify(OUT)
        if command=='run':
            audit.save(OUT/'linear-a-shapes.json',kinship_inventory.inventory(corpus.load()))
            records=[]
            for p in sorted(DAMOS.glob('*.json')):
                item=json.loads(p.read_text()).get('item')
                if item:records.append((p.stem,item))
            audit.save(OUT/'linear-b-markers.json',kinship_inventory.marker_hits(records))
    else:raise ValueError(command)

if __name__=='__main__':main(sys.argv[1])
