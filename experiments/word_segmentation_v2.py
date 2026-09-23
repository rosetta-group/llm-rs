"""Develop and freeze a training-only historical verse word model."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.joint_development_v3 import segmenter
from experiments.segmentation_audit import streams, reference
from experiments.verse_sources import verify as verse
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.segmentation import Segmenter, boundaries
from voynich.verse_word_model import augment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/word-segmentation-v2'
STATE = ROOT / 'artifacts/word-segmentation-v2'
BASE = ROOT / 'artifacts/standard-decipherment/segmenter.json'
PARAMS = ROOT / 'experiments/standard-decipherment/development.json'
CODE = ['voynich/verse_word_model.py', 'voynich/segmentation.py', 'voynich/decipher.py',
        'voynich/corpora.py', 'voynich/data.py', 'experiments/word_segmentation_v2.py',
        'experiments/word_segmentation_fresh.py', 'experiments/segmentation_audit.py',
        'experiments/joint_development_v3.py', 'experiments/verse_sources.py',
        'experiments/historical_sources.py', 'experiments/segmentation.py']


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def grade(prediction, reference):
    if prediction.replace(' ', '') != reference.replace(' ', ''):
        raise ValueError('Letters changed')
    gold, pred = boundaries(reference), boundaries(prediction)
    tp = len(gold & pred); errors = edit_distance(prediction.split(), reference.split())
    return dict(words=len(reference.split()), errors=errors, wer=errors / len(reference.split()),
                tp=tp, fp=len(pred-gold), fn=len(gold-pred),
                precision=tp / len(pred) if pred else float(not gold),
                recall=tp / len(gold) if gold else float(not pred),
                f1=2*tp/(len(pred)+len(gold)) if pred or gold else 1., character_error=0)


def choose(rows):
    base = rows[0]['grades']
    def historical(row): return sum(row['grades'][s]['wer'] for s in ('historical', 'verse')) / 2
    eligible = [r for r in rows[1:] if historical(rows[0])-historical(r) >= .03
                and all(r['grades'][s]['wer']-base[s]['wer'] <= .01 for s in ('historical', 'verse', 'modern'))]
    return min(eligible, key=lambda r:(historical(r),r['grades']['modern']['wer'],r['weight']))['weight'] if eligible else None


def develop():
    if (OUT/'development.json').exists(): raise FileExistsError('Development recorded')
    poems = verse(); base = read(BASE); parameters = read(PARAMS)['selected_segmenter']['parameters']
    refs = {s:reference(t)[1] for s,t in streams().items()}
    rows=[]; train_ids=[]
    for weight in (0,1,4,16):
        model=base
        if weight: model,train_ids=augment(base,poems,weight)
        ws=Segmenter(model,**parameters)
        predictions={s:ws.segment(t.replace(' ','')) for s,t in refs.items()}
        row=dict(weight=weight,grades={s:grade(predictions[s],t) for s,t in refs.items()}, predictions=predictions,
                 lexicon_size=len(model['lexicon']))
        rows.append(row)
        write(OUT/f'dev-weight-{weight}.json',row)
        print(json.dumps(dict(weight=weight,wer={s:g['wer'] for s,g in row['grades'].items()})),flush=True)
    selected=choose(rows)
    if selected:
        model,_=augment(base,poems,selected)
        raw=json.dumps(model,sort_keys=True,separators=(',',':')).encode()
        (OUT/'model.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    write(OUT/'development.json',dict(selected_weight=selected,rows=rows,references=refs,parameters=parameters,
          training_poem_ids=train_ids,base_sha256=digest(BASE),parameters_sha256=digest(PARAMS),
          verse_manifest_sha256=digest(ROOT/'experiments/verse-prior/sources.json'),
          protocol_sha256=digest(OUT/'PROTOCOL.md'),
          code_sha256={p:digest(ROOT/p) for p in CODE if (ROOT/p).exists()},
          development_only=True,voynich_used=False))
    print('Selected weight:',selected,flush=True)


def freeze():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Already frozen')
    dev=read(OUT/'development.json')
    if dev['selected_weight'] is None: raise ValueError('No selected candidate')
    for p,h in dev['code_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('Development code drift: '+p)
    if digest(OUT/'PROTOCOL.md')!=dev['protocol_sha256']: raise ValueError('Protocol drift')
    paths=CODE+['experiments/word-segmentation-v2/PROTOCOL.md','experiments/word-segmentation-v2/development.json',
                'experiments/word-segmentation-v2/model.json.gz','experiments/standard-decipherment/development.json',
                'experiments/standard-decipherment/sources.json','experiments/verse-prior/sources.json',
                'experiments/language-sources.json','experiments/segmentation-sources.json']
    write(OUT/'freeze.json',dict(files={p:digest(ROOT/p) for p in paths},baseline_sha256=digest(BASE),
                               parameters=dev['parameters'],weight=dev['selected_weight'],version=1))
    print('Freeze written; commit before preparing fresh text')


def verify():
    f=read(OUT/'freeze.json')
    for p,h in f['files'].items():
        if digest(ROOT/p)!=h: raise ValueError('Frozen file drift: '+p)
        if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)).hexdigest()!=h:
            raise ValueError('Freeze inputs not committed: '+p)
    p='experiments/word-segmentation-v2/freeze.json'
    if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)).hexdigest()!=digest(ROOT/p):
        raise ValueError('Freeze not committed')
    if digest(BASE)!=f['baseline_sha256']: raise ValueError('Baseline drift')
    return f


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['develop','freeze','verify']);globals()[p.parse_args().command]()
