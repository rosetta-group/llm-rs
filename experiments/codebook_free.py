"""Separated preparation, public-only solving, and hidden-answer grading."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from experiments.segmentation import corpus, check_freeze
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, language_model, edit_distance
from voynich.segmentation import Segmenter
from voynich.unknown_cipher import chunk_counts, solve as decode

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'artifacts/codebook-free'
OUT=ROOT/'experiments/codebook-free'
CODE=['voynich/unknown_cipher.py','experiments/codebook_free.py','voynich/decipher.py','voynich/segmentation.py']


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def hashes(): return {p:digest(ROOT/p) for p in CODE}


def freeze():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Method already frozen')
    segmentation=check_freeze();train,source=corpus('UD_Italian-ISDT','train')
    texts=[normalize(' '.join(r['words'])) for r in train]
    probabilities,counts=language_model(texts,spaces=False)
    STATE.mkdir(parents=True,exist_ok=True)
    np.savez(STATE/'prior.npz',log_probabilities=probabilities,letter_counts=counts)
    write(STATE/'chunks.json',chunk_counts(texts))
    frozen=dict(code_sha256=hashes(),protocol_sha256=digest(OUT/'PROTOCOL.md'),
        prior_sha256=digest(STATE/'prior.npz'),chunks_sha256=digest(STATE/'chunks.json'),prior_source=source,
        segmentation_freeze_sha256=digest(ROOT/'artifacts/segmentation/frozen.json'),
        restarts=6,steps=12000,cap_seconds=120,seed=42,at=time.time())
    write(OUT/'freeze.json',frozen)


def verify():
    frozen=read(OUT/'freeze.json');check_freeze()
    if frozen['code_sha256']!=hashes(): raise ValueError('Code drift')
    for key,path in [('protocol_sha256',OUT/'PROTOCOL.md'),('prior_sha256',STATE/'prior.npz'),
                     ('chunks_sha256',STATE/'chunks.json'),('segmentation_freeze_sha256',ROOT/'artifacts/segmentation/frozen.json')]:
        if frozen[key]!=digest(path): raise ValueError('Frozen input drift')
    return frozen


def artificial(text,rng):
    mapping={};used=set();tokens=[];i=0
    while i<len(text):
        size=rng.choice([1,2]);chunk=text[i:i+size];i+=len(chunk)
        variant=(chunk,rng.randrange(2))
        if variant not in mapping:
            while True:
                token=''.join(rng.choices('abcdefghilmnopqrstuvxyz',k=4))
                if token not in used: break
            used.add(token);mapping[variant]=token
        tokens.append(mapping[variant])
    return ' '.join(tokens), {token:chunk for (chunk,_),token in mapping.items()}


def prepare():
    frozen=verify()
    if (STATE/'public.json').exists(): raise FileExistsError('Challenge already generated')
    prior=set()
    for split in ('train','dev'):
        rows,_=corpus('UD_Italian-ISDT',split)
        prior.update(normalize(' '.join(r['words'])) for r in rows)
    excluded={'historical':set(),'modern':set()}
    for r in read(ROOT/'artifacts/decipherment/evaluator-only/answers.json'):
        excluded['historical'].update(r['source_sentences'])
    for r in read(ROOT/'artifacts/segmentation/evaluator-only/answers.json'):
        excluded[r['dataset']].update(r['source_ids'])
    passages=[]
    for dataset,repo,split in [('modern','UD_Italian-ISDT','test'),('historical','UD_Italian-Old','train')]:
        rows,source=corpus(repo,split)
        for region in range(2):
            texts=[];ids=[];length=0
            for row in rows[region*len(rows)//2:(region+1)*len(rows)//2]:
                text=normalize(' '.join(row['words']));n=len(text.replace(' ',''))
                if row['id'] in excluded[dataset] or text in prior or not text or n>1800:
                    if length<1200: texts=[];ids=[];length=0
                    else: break
                    continue
                if length+n>1800:
                    if length>=1200: break
                    texts=[];ids=[];length=0
                texts.append(text);ids.append(row['id']);length+=n
                if length>=1200: break
            if length<1200: raise ValueError('Not enough fresh contiguous text')
            if set(ids)&excluded[dataset]: raise ValueError('Reused sentence')
            excluded[dataset].update(ids)
            passages.append(dict(dataset=dataset,source=source,source_ids=ids,plaintext=' '.join(texts),
                                 passage=hashlib.sha256((repo+':'.join(ids)).encode()).hexdigest()[:16]))
    vendor_dir=ROOT/'artifacts/decipherment/vendor'
    for f in read(ROOT/'experiments/decipherment-sources.json')['files']:
        if digest(ROOT/f['path'])!=f['sha256']: raise ValueError('Encoder source drift')
    spec=importlib.util.spec_from_file_location('published_naibbe',vendor_dir/'naibbe.py')
    vendor=importlib.util.module_from_spec(spec);cwd=os.getcwd()
    try:
        os.chdir(vendor_dir);spec.loader.exec_module(vendor)
    finally: os.chdir(cwd)
    public=[];answers=[]
    for passage in passages:
        dense=passage['plaintext'].replace(' ','')
        for family in ('substitution','variable-homophonic','naibbe'):
            seed=random.SystemRandom().randrange(2**63);rng=random.Random(seed)
            alphabet=list(ALPHABET);rng.shuffle(alphabet);key=dict(zip(ALPHABET,alphabet))
            inverse={v:k for k,v in key.items()};coded=''.join(key[c] for c in dense)
            if family=='substitution': ciphertext=coded;roundtrip=''.join(inverse[c] for c in ciphertext)
            elif family=='variable-homophonic':
                ciphertext,mapping=artificial(dense,rng)
                roundtrip=''.join(mapping[t] for t in ciphertext.split())
            else:
                random.seed(seed);trace=io.StringIO()
                tokens=vendor.encrypt_naibbe(coded,vendor.naibbe_tables,vendor.placeholder_to_glyph,use_78=False,pre_plaintext_file=trace)
                ciphertext=' '.join(tokens);roundtrip=''.join(inverse[c] for c in ''.join(trace.getvalue().split()))
            if roundtrip!=dense: raise ValueError('Encoder roundtrip failed')
            ident=hashlib.sha256((ciphertext+str(seed)).encode()).hexdigest()[:16]
            public.append(dict(id=ident,ciphertext=ciphertext))
            answers.append(dict(passage,id=ident,family=family,seed=seed,encryption_key=key,roundtrip=True))
    private=STATE/'evaluator-only';private.mkdir(mode=0o700,exist_ok=True)
    write(STATE/'public.json',sorted(public,key=lambda r:r['id']));write(private/'answers.json',answers)
    write(STATE/'challenge.json',dict(public_sha256=digest(STATE/'public.json'),answers_sha256=digest(private/'answers.json'),
        freeze_sha256=digest(OUT/'freeze.json'),cases=len(public),passages=len(passages),old_sentence_overlap=0,prior_sentence_overlap=0,
        sources=[p['source'] for p in passages],at=time.time()))
    print('Prepared 12 cases from four fresh passages; references stay evaluator-only',flush=True)


def solve():
    frozen=verify()
    if (STATE/'predictions.json').exists(): raise FileExistsError('Predictions already frozen')
    prior=dict(np.load(STATE/'prior.npz'));prior['chunks']=read(STATE/'chunks.json')
    segmenter=Segmenter(read(ROOT/'artifacts/segmentation/model.json'),**read(ROOT/'artifacts/segmentation/frozen.json')['parameters'])
    results=[];started=time.time();public=read(STATE/'public.json')
    for i,case in enumerate(public):
        result=decode(case['ciphertext'],prior,seed=frozen['seed']+i,restarts=frozen['restarts'],
                      steps=frozen['steps'],cap=frozen['cap_seconds'])
        result['segmented']=segmenter.segment(result['recovered']);result['id']=case['id'];results.append(result)
        write(STATE/'progress.json',dict(completed=len(results),total=len(public),seconds=time.time()-started))
        print(f"Recovered {len(results)}/{len(public)} opaque cases ({time.time()-started:.1f}s)",flush=True)
    write(STATE/'predictions.json',dict(results=results,seconds=time.time()-started,code_sha256=hashes(),
        public_sha256=digest(STATE/'public.json'),freeze_sha256=digest(OUT/'freeze.json'),at=time.time(),
        access='Ciphertext, unpaired Italian prior, frozen segmenter only; no family, codebook, answers, traces or lengths'))


def evaluate():
    verify();pred=read(STATE/'predictions.json');challenge=read(STATE/'challenge.json')
    for key,path in [('public_sha256',STATE/'public.json'),('answers_sha256',STATE/'evaluator-only/answers.json'),('freeze_sha256',OUT/'freeze.json')]:
        if challenge[key]!=digest(path): raise ValueError('Challenge drift')
    if pred['code_sha256']!=hashes() or pred['public_sha256']!=challenge['public_sha256'] or pred['freeze_sha256']!=challenge['freeze_sha256']:
        raise ValueError('Prediction provenance drift')
    answers={r['id']:r for r in read(STATE/'evaluator-only/answers.json')}
    if len(pred['results'])!=len(answers) or {r['id'] for r in pred['results']}!=set(answers): raise ValueError('Incomplete predictions')
    rows=[]
    for result in pred['results']:
        gold=answers[result['id']];plain=gold['plaintext'];dense=plain.replace(' ','')
        character_errors=edit_distance(result['recovered'],dense);word_errors=edit_distance(result['segmented'].split(),plain.split())
        candidates=[dict(representation=c['representation'],method=c['method'],language_score=c['language_score'],
            character_error_rate=edit_distance(c['recovered'],dense)/len(dense),proposals=c['proposals'],cap_hit=c['cap_hit'],seconds=c['seconds']) for c in result['candidates']]
        cer=character_errors/len(dense);wer=word_errors/len(plain.split())
        rows.append(dict(id=result['id'],passage=gold['passage'],dataset=gold['dataset'],family=gold['family'],
            characters=len(dense),words=len(plain.split()),character_errors=character_errors,word_errors=word_errors,
            character_error_rate=cer,word_error_rate=wer,recovery_gate=cer<=.01 and wer<=.1,
            exact=result['segmented']==plain,selected=result['selected'],candidates=candidates,
            oracle_candidate_cer=min(c['character_error_rate'] for c in candidates)))
    summary={}
    for family in ('substitution','variable-homophonic','naibbe'):
        summary[family]={}
        for dataset in ('modern','historical'):
            subset=[r for r in rows if r['family']==family and r['dataset']==dataset]
            summary[family][dataset]=dict(cases=len(subset),character_error_rate=sum(r['character_errors'] for r in subset)/sum(r['characters'] for r in subset),
                word_error_rate=sum(r['word_errors'] for r in subset)/sum(r['words'] for r in subset),
                recovery_gate_passes=sum(r['recovery_gate'] for r in subset),exact=sum(r['exact'] for r in subset))
    write(OUT/'results.json',dict(summary=summary,cases=rows,seconds=pred['seconds'],challenge=challenge,
        predictions_sha256=digest(STATE/'predictions.json'),freeze=read(OUT/'freeze.json'),test_scored=False,
        limitation='Four independent passages; broader mapping search is not validated unless its positive control succeeds. No claim about all unknown ciphers.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','prepare','solve','evaluate']);globals()[p.parse_args().command]()
