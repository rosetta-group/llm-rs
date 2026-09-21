"""Frozen broad-domain classification, nuisance checks, and grouped uncertainty."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import time
import numpy as np

from voynich.data import digest
from voynich.association_complex import holm
from voynich.image_domains import aggregate,operators,score,predictions,shuffle,movable

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/image-domains'
STATE=ROOT/'artifacts/image-domains'
FILES=['voynich/image_domains.py','experiments/image_domains.py','voynich/data.py',
       'voynich/association_complex.py','experiments/image-domains/PROTOCOL.md',
       'artifacts/data/gc/corpus.json','artifacts/data/gc/manifest.json']


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def dataset():
    return aggregate(read(ROOT/'artifacts/data/gc/corpus.json'),read(ROOT/'artifacts/data/gc/manifest.json')['assignments'])


def audit(rows,coverage):
    domains=sorted({r['domain'] for r in rows})
    tables={name:{d:dict(Counter(r[name] for r in rows if r['domain']==d)) for d in domains} for name in ('hand_signature','quire')}
    labels=np.array([r['domain'] for r in rows])
    return dict(coverage,domains={d:dict(folios=sum(r['domain']==d for r in rows),pages=sum(len(r['pages']) for r in rows if r['domain']==d)) for d in domains},
        tables=tables,hand_mobility=movable(labels,[r['hand_signature'] for r in rows]),
        hand_quire_mobility=movable(labels,[r['hand_signature']+'|'+r['quire'] for r in rows]))


def freeze():
    if (OUT/'freeze.json').exists():raise FileExistsError('Domain study already frozen')
    rows,coverage=dataset()
    record=dict(files_sha256={p:digest(ROOT/p) for p in FILES},audit=audit(rows,coverage),
        seed=20260922,permutations=999,bootstrap=2000,cpu_cap_seconds=3600,output_cap_mib=100,
        source=dict(transcription='GC2a-n, v101; IVTFF 2a modified 2025-06-25',
                    illustration_types='https://www.voynich.nu/software/ivtt/IVTFF_format.pdf',
                    visual_overview='https://www.voynich.nu/illustr.html'),frozen_at=time.time())
    write(OUT/'freeze.json',record);print(json.dumps(record['audit'],indent=2))


def verify():
    frozen=read(OUT/'freeze.json')
    for path,expected in frozen['files_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen input drift: '+path)
    return frozen


def bootstrap(y,base,candidate,groups,classes,draws,rng):
    values=[];groups=np.asarray(groups);unique=sorted(set(groups))
    for _ in range(draws):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        if len(set(y[ix]))!=classes:continue
        values.append(score(y[ix],candidate[ix],classes)['macro_recall']-score(y[ix],base[ix],classes)['macro_recall'])
    return dict(interval_95=np.quantile(values,[.025,.975]).tolist(),valid_draws=len(values),omitted_draws=draws-len(values))


def evaluate_design(rows,group_key,frozen,rng):
    classes=sorted({r['domain'] for r in rows});k=len(classes);y=np.array([classes.index(r['domain']) for r in rows])
    groups=np.array([r[group_key] for r in rows]);strata=np.array([r['hand_signature'] for r in rows])
    for group in set(groups):
        if len(set(y[groups!=group]))!=k:raise ValueError('A training fold lacks a class')
    maps=operators(rows,groups);guesses={name:predictions(a,y,k) for name,a in maps.items()}
    metrics={name:score(y,p,k) for name,p in guesses.items()};mobility=movable(y,strata)
    nulls={view:[] for view in ('words','characters')}
    if mobility['movable_folios']:
        for _ in range(frozen['permutations']):
            yp=shuffle(y,strata,rng)
            baseline=score(yp,predictions(maps['controls'],yp,k),k)['macro_recall']
            for view in nulls:
                nulls[view].append(score(yp,predictions(maps[view],yp,k),k)['macro_recall']-baseline)
    tests={}
    for view,null in nulls.items():
        gain=metrics[view]['macro_recall']-metrics['controls']['macro_recall']
        p=(1+sum(value>=gain-1e-12 for value in null))/(len(null)+1) if null else None
        tests[view]=dict(gain=gain,p=p,conditional_test='identifiable under hand-stratified null' if null else 'unidentifiable: no labels move within hand strata',
            **bootstrap(y,guesses['controls'],guesses[view],groups,k,frozen['bootstrap'],rng))
    return dict(classes=classes,n=len(rows),groups=len(set(groups)),group_key=group_key,
        metrics=metrics,tests=tests,null_gains=nulls,hand_mobility=mobility,
        predictions=[dict(folio=r['folio'],pages=r['pages'],domain=r['domain'],hand_signature=r['hand_signature'],quire=r['quire'],
            predictions={name:classes[int(p[i])] for name,p in guesses.items()}) for i,r in enumerate(rows)])


def sanity():
    rows=[]
    for i in range(18):
        tokens=[['aaaa','abab'],['mmmm','mnmn'],['xxxx','xyxy']][i%3]*10
        rows.append(dict(tokens=tokens,loci=10,pages=['p'],unknown=0,hands={'1':10},kinds={'P':10},position=i,quire=str(i//3)))
    maps=operators(rows,np.arange(18));y=np.arange(18)%3
    return dict(examples=18,task='Three planted vocabularies with identical layout and hand',
        scores={name:score(y,predictions(maps[name],y,3),3) for name in ('controls','words','characters')})


def run():
    frozen=verify()
    if (OUT/'results.json').exists():raise FileExistsError('Domain study already graded')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Disk reserve')
    start=time.monotonic();rng=np.random.default_rng(frozen['seed']);rows,coverage=dataset()
    designs={};designs['folio']=evaluate_design(rows,'folio',frozen,rng)
    eligible={domain for domain in {r['domain'] for r in rows}
              if len({r['quire'] for r in rows if r['domain']==domain})>=2 and sum(r['domain']==domain for r in rows)>=5}
    cross=[r for r in rows if r['domain'] in eligible]
    designs['quire']=evaluate_design(cross,'quire',frozen,rng)
    tests={name+'/'+view:record for name,design in designs.items() for view,record in design['tests'].items()}
    correction=holm({key:value['p'] if value['p'] is not None else 1. for key,value in tests.items()})
    for key,value in tests.items():
        value['holm_p']=correction[key] if value['p'] is not None else None
        value['supported_increment']=bool(value['p'] is not None and value['gain']>0 and correction[key]<=.05 and value['interval_95'][0]>0)
    result=dict(audit=frozen['audit'],designs=designs,tests=tests,
        excluded_cross_quire_domains=sorted({r['domain'] for r in rows}-eligible),sanity=sanity(),
        freeze_sha256=digest(OUT/'freeze.json'),seconds=time.monotonic()-start,final_test_scored=False,
        source_scope='Conventional illustration-domain metadata; not independently re-annotated pixels',
        inference='Association with broad visual categories; not a word meaning or proof of image semantics independent of book structure')
    if result['seconds']>frozen['cpu_cap_seconds']:raise TimeoutError('CPU cap exceeded')
    write(OUT/'results.json',result)
    print(json.dumps(dict(metrics={name:d['metrics'] for name,d in designs.items()},tests=tests,seconds=result['seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run','verify'])
    result=globals()[parser.parse_args().command]()
    if result is not None:print('Frozen domain inputs verified')
