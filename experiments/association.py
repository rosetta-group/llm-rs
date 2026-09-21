"""Frozen, folio-held-out association pilot using existing visual descriptions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np

from voynich.data import digest

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/association'
OUT = ROOT / 'experiments/association'
URL = 'https://www.ic.unicamp.br/~stolfi/EXPORT/voynich/98-02-01-lotsa-labels/labtit-98-07-20.idx'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def parse(text):
    chosen = {}
    for line in text.splitlines():
        fields = [f.strip() for f in line.split('|')]
        if len(fields) != 11 or fields[8] != 'P':
            continue
        _, section, page, group, index, transcriber, label, _, _, obj, description = fields
        key = (page, group, index)
        rank = {'V': 0, 'C': 1, 'L': 2, 'F': 3}.get(transcriber, 4)
        row = dict(page=page, group=group, index=int(index), transcriber=transcriber,
                   label=label, object=obj, description=description, section=section, rank=rank)
        if key not in chosen or rank < chosen[key]['rank']:
            chosen[key] = row
    return list(chosen.values())


def endpoint(description):
    if '?' in description:
        return None
    colors = set(re.findall(r'\b(light|dark)(?: colo[u]?red)? roots?\b', description.lower()))
    return int('dark' in colors) if len(colors) == 1 else None


def eligible(rows, metadata, assignments):
    maxima = defaultdict(int)
    for r in rows:
        maxima[r['page'], r['group']] = max(maxima[r['page'], r['group']], r['index'])
    selected = []
    for r in rows:
        page = re.sub(r'^f101v[12]$', 'f101v', r['page'])
        meta = metadata.get(page)
        target = endpoint(r['description'])
        if (meta is None or assignments.get(meta['folio']) != 'train' or target is None
                or r['object'] not in ('plant', 'root') or not re.fullmatch('[a-z]+(?:\.[a-z]+)*', r['label'])):
            continue
        selected.append(dict(r, folio=meta['folio'], hand=meta['hand'], target=target,
                             position=r['index'] / max(1, maxima[r['page'], r['group']])))
    return selected


def features(rows):
    groups = sorted({r['group'] for r in rows}); readers = sorted({r['transcriber'] for r in rows})
    controls = np.array([[len(r['label'].replace('.', '')), len(r['label'].split('.')), r['position']]
                        + [float(r['group'] == g) for g in groups]
                        + [float(r['transcriber'] == t) for t in readers] for r in rows])
    grams = np.zeros((len(rows), 256))
    for i, r in enumerate(rows):
        label = r['label']
        for n in (1, 2, 3):
            for j in range(len(label) - n + 1):
                bucket = int.from_bytes(hashlib.blake2b(label[j:j+n].encode(), digest_size=4).digest(), 'big') % 256
                grams[i, bucket] += 1
        grams[i] /= max(1, grams[i].sum())
    return controls, np.column_stack([controls, grams])


def prediction_operator(x, groups, penalty=10.):
    """OOF ridge map: applying this matrix to y equals refitting every held-out fold."""
    groups = np.asarray(groups)
    result = np.zeros((len(x), len(x)))
    for group in sorted(set(groups)):
        train, test = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
        mean = x[train].mean(axis=0); scale = x[train].std(axis=0)
        scale[scale < 1e-12] = 1
        a, b = (x[train]-mean)/scale, (x[test]-mean)/scale
        # Centering y gives an unpenalized intercept. Dual form is small for this pilot.
        center = np.eye(len(train)) - np.ones((len(train), len(train))) / len(train)
        result[np.ix_(test, train)] = b @ a.T @ np.linalg.solve(a @ a.T + penalty*np.eye(len(train)), center) + 1/len(train)
    return result


def balanced_accuracy(y, predicted):
    if len(set(y)) != 2:
        return None
    return float(np.mean([np.mean(predicted[y == c] == c) for c in (0, 1)]))


def permute_within(y, pages, rng):
    result = y.copy()
    for page in sorted(set(pages)):
        indices = np.flatnonzero(pages == page)
        result[indices] = rng.permutation(y[indices])
    return result


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Association pilot already frozen')
    STATE.mkdir(exist_ok=True, parents=True)
    if not (STATE / 'labels.idx').exists():
        urllib.request.urlretrieve(URL, STATE / 'labels.idx')
    write(OUT / 'freeze.json', dict(protocol_sha256=digest(OUT / 'PROTOCOL.md'), code_sha256=digest(__file__),
        source=dict(url=URL, path='artifacts/association/labels.idx', sha256=digest(STATE/'labels.idx'),
                    attribution='John Grove / Jorge Stolfi, 1998', reuse_license='Not found; raw copy remains local'),
        metadata_sha256=digest(ROOT/'artifacts/data/gc/documents.json'),
        split_sha256=digest(ROOT/'artifacts/data/gc/manifest.json'), seed=42, permutations=999, bootstrap=2000))


def run():
    frozen = read(OUT/'freeze.json')
    if (OUT/'results.json').exists(): raise FileExistsError('Pilot already graded')
    for expected, path in [(frozen['code_sha256'], Path(__file__)), (frozen['protocol_sha256'],OUT/'PROTOCOL.md'),
                           (frozen['source']['sha256'], STATE/'labels.idx'),
                           (frozen['metadata_sha256'],ROOT/'artifacts/data/gc/documents.json'),
                           (frozen['split_sha256'],ROOT/'artifacts/data/gc/manifest.json')]:
        if expected != digest(path): raise ValueError('Frozen pilot input changed')
    rows = parse((STATE/'labels.idx').read_text())
    metadata = {r['page']: {k:r[k] for k in ('folio','hand')} for r in read(ROOT/'artifacts/data/gc/documents.json')}
    assignments = read(ROOT/'artifacts/data/gc/manifest.json')['assignments']
    selected = eligible(rows, metadata, assignments)
    counts = {str(c): dict(n=sum(r['target']==c for r in selected), folios=sorted({r['folio'] for r in selected if r['target']==c})) for c in (0,1)}
    audit = dict(deduplicated_objects=len(rows), object_classes=dict(Counter(r['object'] for r in rows)),
                 eligible=len(selected), classes=counts, hands=sorted({r['hand'] for r in selected}),
                 sections=sorted({r['section'] for r in selected}))
    if any(r['hand'] != '1' or r['section'] != 'pharma' for r in selected):
        # Catalogue uses its own section spelling; require one fixed section rather than pool sections.
        if len(audit['hands']) != 1 or len(audit['sections']) != 1 or audit['hands'] != ['1']:
            raise ValueError('Hand/section assumption failed')
    if any(v['n'] < 20 or len(v['folios']) < 4 for v in counts.values()):
        write(OUT/'results.json',dict(audit=audit, status='ineligible', freeze=frozen)); print(audit); return
    y=np.array([r['target'] for r in selected]); folios=np.array([r['folio'] for r in selected]); pages=np.array([r['page'] for r in selected])
    controls, full = features(selected)
    maps = [prediction_operator(x, folios) for x in (controls,full)]
    guesses = [(a@y >= .5).astype(int) for a in maps]
    scores = [balanced_accuracy(y,p) for p in guesses]; gain=scores[1]-scores[0]
    rng=np.random.default_rng(frozen['seed']); null=[]
    for _ in range(frozen['permutations']):
        yp=permute_within(y,pages,rng)
        values=[balanced_accuracy(yp,(a@yp >= .5).astype(int)) for a in maps]
        null.append(values[1]-values[0])
    unique=sorted(set(folios)); boot=[]
    for _ in range(frozen['bootstrap']):
        indices=np.concatenate([np.flatnonzero(folios==g) for g in rng.choice(unique,len(unique),replace=True)])
        if len(set(y[indices]))==2:
            boot.append(balanced_accuracy(y[indices],guesses[1][indices])-balanced_accuracy(y[indices],guesses[0][indices]))
    interval=np.quantile(boot,[.025,.975]).tolist(); p=(1+sum(v>=gain-1e-12 for v in null))/(len(null)+1)
    summary=dict(control_balanced_accuracy=scores[0], text_balanced_accuracy=scores[1], gain=gain,
                 permutation_p=p, folio_bootstrap_interval_95=interval, promising=bool(gain>0 and p<=.05 and interval[0]>0))
    per_folio=[]
    for g in unique:
        ix=folios==g
        per_folio.append(dict(folio=g,n=int(ix.sum()),dark=int(y[ix].sum()),
            control_accuracy=float(np.mean(guesses[0][ix]==y[ix])),text_accuracy=float(np.mean(guesses[1][ix]==y[ix]))))
    write(OUT/'results.json',dict(status='completed', audit=audit,summary=summary, per_folio=per_folio,
          null_gains=null, freeze=frozen, test_scored=False,
          limitations=['Small pharmaceutical-label sample, not herbal-page coverage', 'Annotations not made blind to text',
                       'Missing descriptions excluded, not negative labels', 'Within-page null controls page composition, not all layout effects']))
    print(json.dumps(dict(audit=audit,summary=summary,per_folio=per_folio),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);globals()[p.parse_args().command]()
