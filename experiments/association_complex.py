"""Freeze and run the exploratory nonlinear, joint-profile association extension."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.association import read,write,parse,eligible,balanced_accuracy,permute_within
from voynich.association_complex import (BASE,NAMES,select_profiles,features,operators,
    target_folds,profile_scores,holm)
from voynich.data import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/association-complex'
STATE=ROOT/'artifacts/association-complex'
MODELS=('additive','interactions','nonlinear')
METRICS=('root_color','profile_error','matching_rank','relational_alignment')
CODE=['voynich/association_complex.py','experiments/association_complex.py','experiments/association.py']


def dataset():
    source=read(ROOT/'experiments/association/freeze.json')
    if digest(ROOT/source['source']['path'])!=source['source']['sha256']:raise ValueError('Annotation source drift')
    metadata={r['page']:{k:r[k] for k in ('folio','hand')} for r in read(ROOT/'artifacts/data/gc/documents.json')}
    assignments=read(ROOT/'artifacts/data/gc/manifest.json')['assignments']
    rows=parse((ROOT/source['source']['path']).read_text())
    return eligible(rows,metadata,assignments),select_profiles(rows,metadata,assignments)


def freeze():
    if (OUT/'freeze.json').exists():raise FileExistsError('Complex study already frozen')
    binary,profiles=dataset();counts=Counter(r['page'] for r in profiles)
    audit=dict(binary_objects=len(binary),profile_objects=len(profiles),
        profile_folios=dict(Counter(r['folio'] for r in profiles)),
        matching_objects=sum(n for n in counts.values() if n>=3),
        descriptor_mentions={name:int(sum(r['profile'][i] for r in profiles)) for i,name in enumerate(BASE)},
        profile_eligible=len(profiles)>=60 and len({r['folio'] for r in profiles})>=4 and sum(n for n in counts.values() if n>=3)>=40)
    files=CODE+['experiments/association-complex/PROTOCOL.md','experiments/association/freeze.json',
                'artifacts/association/labels.idx','artifacts/data/gc/documents.json','artifacts/data/gc/manifest.json']
    record=dict(files_sha256={p:digest(ROOT/p) for p in files},audit=audit,seed=20260921,
                permutations=999,bootstrap=2000,planned_tests=12,penalty=1.,cpu_cap_seconds=3600,
                output_cap_bytes=100*1024**2,frozen_at=time.time(),status='exploratory reuse; not independent confirmation')
    write(OUT/'freeze.json',record);print(json.dumps(audit,indent=2))


def verify():
    frozen=read(OUT/'freeze.json')
    for path,expected in frozen['files_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen input changed: '+path)
    return frozen


def profile_summary(scores):
    return dict(profile_error=float(scores['mse'].mean()),matching_rank=float(np.nanmean(scores['matching'])),
                relational_alignment=float(np.mean([value for _,value in scores['relations']])))


def improvement(model,baseline,metric):
    return baseline-model if metric=='profile_error' else model-baseline


def profile_bootstrap(baseline,candidate,folios,draws,rng):
    unique=sorted(set(folios));result={k:[] for k in METRICS[1:]}
    for _ in range(draws):
        groups=rng.choice(unique,len(unique),replace=True)
        ix=np.concatenate([np.flatnonzero(folios==g) for g in groups])
        result['profile_error'].append(float((baseline['mse'][ix]-candidate['mse'][ix]).mean()))
        valid=np.isfinite(baseline['matching'][ix])
        if valid.any():result['matching_rank'].append(float((candidate['matching'][ix][valid]-baseline['matching'][ix][valid]).mean()))
        a=[v for g in groups for f,v in baseline['relations'] if f==g]
        b=[v for g in groups for f,v in candidate['relations'] if f==g]
        if a:result['relational_alignment'].append(float(np.mean(b)-np.mean(a)))
    return {k:np.quantile(v,[.025,.975]).tolist() for k,v in result.items()}


def sanity(permutations,seed):
    x=np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]])/np.sqrt(2),(6,1))
    y=(x[:,0]*x[:,1]>0).astype(int);groups=np.repeat(np.arange(6),4)
    maps=operators(np.zeros((24,1)),x,groups);rng=np.random.default_rng(seed)
    # Numerical roundoff at a 0.5 tie is defined as a positive prediction throughout this study.
    score=lambda p:balanced_accuracy(y,p>=.5-1e-12)
    observed={name:score(a@y) for name,a in maps.items()};null={name:[] for name in MODELS}
    for _ in range(permutations):
        yp=permute_within(y,groups,rng)
        values={name:balanced_accuracy(yp,a@yp>=.5-1e-12) for name,a in maps.items()}
        for name in MODELS:null[name].append(values[name]-values['control'])
    return dict(examples=24,groups=6,task='XOR interaction with constant length/layout controls',
        balanced_accuracy=observed,permutation_p={name:(1+sum(v>=observed[name]-observed['control']-1e-12 for v in null[name]))/(permutations+1) for name in MODELS})


def run():
    frozen=verify()
    if (OUT/'results.json').exists():raise FileExistsError('Complex study already scored')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Disk reserve')
    STATE.mkdir(parents=True,exist_ok=True);started=time.monotonic();rng=np.random.default_rng(frozen['seed'])
    rows,profiles=dataset();tests={};details={};nulls={};B=frozen['permutations']
    def budget(stage,completed):
        elapsed=time.monotonic()-started
        write(STATE/'progress.json',dict(stage=stage,completed=completed,permutations=B,seconds=elapsed))
        if elapsed>frozen['cpu_cap_seconds']:raise TimeoutError('Frozen CPU cap exceeded; incomplete run preserved')
    y=np.array([r['target'] for r in rows]);folios=np.array([r['folio'] for r in rows]);pages=np.array([r['page'] for r in rows])
    maps=operators(*features(rows),folios)
    predictions={name:a@y for name,a in maps.items()}
    guesses={name:v>=.5-1e-12 for name,v in predictions.items()}
    observed={name:balanced_accuracy(y,v) for name,v in guesses.items()}
    for name in MODELS:
        key=name+'/root_color';tests[key]=dict(model=name,endpoint='root_color',baseline=observed['control'],score=observed[name],gain=observed[name]-observed['control']);nulls[key]=[]
    for iteration in range(B):
        yp=permute_within(y,pages,rng)
        scores={name:balanced_accuracy(yp,a@yp>=.5-1e-12) for name,a in maps.items()}
        for name in MODELS:nulls[name+'/root_color'].append(scores[name]-scores['control'])
        if iteration%100==0:budget('binary permutations',iteration)
    for name in MODELS:
        boot=[];unique=sorted(set(folios))
        for _ in range(frozen['bootstrap']):
            ix=np.concatenate([np.flatnonzero(folios==g) for g in rng.choice(unique,len(unique),replace=True)])
            if len(set(y[ix]))==2:boot.append(balanced_accuracy(y[ix],guesses[name][ix])-balanced_accuracy(y[ix],guesses['control'][ix]))
        tests[name+'/root_color']['interval_95']=np.quantile(boot,[.025,.975]).tolist()
    write(STATE/'binary-predictions.json',dict(locations=[f"{r['page']}.{r['group']}.{r['index']}" for r in rows],
          scores={name:v.tolist() for name,v in predictions.items()}))
    details['binary']=dict(n=len(rows),folios=sorted(set(folios)),scores=observed)
    if frozen['audit']['profile_eligible']:
        y=np.array([r['profile'] for r in profiles]);folios=np.array([r['folio'] for r in profiles]);pages=np.array([r['page'] for r in profiles])
        maps=operators(*features(profiles),folios);folds=target_folds(y,folios,pages)
        predictions={name:a@y for name,a in maps.items()}
        outputs={name:profile_scores(p,y,folds) for name,p in predictions.items()}
        observed={name:profile_summary(v) for name,v in outputs.items()}
        for name in MODELS:
            intervals=profile_bootstrap(outputs['control'],outputs[name],folios,frozen['bootstrap'],rng)
            for metric in METRICS[1:]:
                key=name+'/'+metric;base=observed['control'][metric];value=observed[name][metric]
                tests[key]=dict(model=name,endpoint=metric,baseline=base,score=value,gain=improvement(value,base,metric),interval_95=intervals[metric]);nulls[key]=[]
        for iteration in range(B):
            yp=permute_within(y,pages,rng)
            scores={name:profile_summary(profile_scores(a@yp,yp,folds)) for name,a in maps.items()}
            for name in MODELS:
                for metric in METRICS[1:]:nulls[name+'/'+metric].append(improvement(scores[name][metric],scores['control'][metric],metric))
            if iteration%100==0:budget('joint-profile permutations',iteration)
        details['profiles']=dict(n=len(profiles),scores=observed,
            folds=[dict(folio=f['folio'],n=len(f['test']),active_targets=len(f['active']),
                        targets=[NAMES[i] for i in f['active']]) for f in folds],
            per_folio={str(g):{name:dict(profile_error=float(v['mse'][folios==g].mean()),
                         matching_rank=float(np.nanmean(v['matching'][folios==g])) if np.isfinite(v['matching'][folios==g]).any() else None)
                         for name,v in outputs.items()} for g in sorted(set(folios))})
        np.savez_compressed(STATE/'profile-predictions.npz',targets=y,**predictions)
    else:
        for name in MODELS:
            for metric in METRICS[1:]:tests[name+'/'+metric]=dict(model=name,endpoint=metric,status='infeasible',p=1.)
    for key,row in tests.items():
        if key in nulls:row['p']=(1+sum(v>=row['gain']-1e-12 for v in nulls[key]))/(B+1)
    adjusted=holm({key:r['p'] for key,r in tests.items()})
    for key,row in tests.items():
        row['holm_p']=adjusted[key]
        row['signal']=bool(row.get('gain',0)>0 and row['holm_p']<=.05 and row.get('interval_95',[0])[0]>0)
    check=sanity(B,frozen['seed']);budget('complete',B)
    result=dict(exploratory=True,tests=tests,audit=frozen['audit'],details=details,sanity=check,
        null_gains=nulls,seconds=time.monotonic()-started,freeze_sha256=digest(OUT/'freeze.json'),
        test_scored=False,limitations=['Reused source and pilot examples; not independent confirmation',
        'Human descriptions, not raw pixels; annotators could see writing','Missing feature mentions are unknown, not absence',
        'Six folios; shape and texture labels are sparse','Fixed kernels; failed tests do not rule out all nonlinear associations'])
    write(OUT/'results.json',result)
    size=sum(p.stat().st_size for folder in (OUT,STATE) for p in folder.rglob('*') if p.is_file())
    if size>frozen['output_cap_bytes']:raise RuntimeError('Output cap exceeded')
    print(json.dumps(dict(tests=tests,sanity=check,seconds=result['seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run','verify'])
    result=globals()[parser.parse_args().command]()
    if result is not None:print('Frozen inputs verified')
