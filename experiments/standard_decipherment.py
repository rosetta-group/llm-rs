"""Develop, commit-freeze, solve ciphertext only, then grade fresh Italian controls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from experiments.historical_sources import verify as historical
from experiments.segmentation import corpus, check_freeze
from experiments.codebook_free import artificial
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, language_model, edit_distance
from voynich.description_length import CharacterPrior, description_length, infer_mapping
from voynich.homophonic import beam_search, hmm_em
from voynich.segmentation import fit, Segmenter
from voynich.unknown_cipher import solve as legacy_solve, chunk_counts, representations

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'artifacts/standard-decipherment'
OUT = ROOT/'experiments/standard-decipherment'
CODE = ['voynich/description_length.py', 'voynich/homophonic.py', 'experiments/standard_decipherment.py',
        'experiments/historical_sources.py', 'voynich/segmentation.py', 'voynich/unknown_cipher.py',
        'voynich/decipher.py', 'experiments/segmentation.py', 'experiments/codebook_free.py',
        'voynich/corpora.py', 'voynich/data.py']


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def hashes(): return {p:digest(ROOT/p) for p in CODE}


def windows(text, words=80):
    values = text.split()
    return [' '.join(values[i:i+words]) for i in range(0,len(values)-words+1,words)]


def homophonic(text, rng):
    labels = [f'G{i:03}' for i in range(2*len(ALPHABET))]
    rng.shuffle(labels)
    key = {labels[2*i+j]:c for i,c in enumerate(ALPHABET) for j in range(2)}
    forward = {c:[s for s,v in key.items() if v==c] for c in ALPHABET}
    tokens = [rng.choice(forward[c]) for c in text]
    return ' '.join(tokens), {s:key[s] for s in set(tokens)}


def develop():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Method frozen')
    STATE.mkdir(parents=True,exist_ok=True)
    stories = historical()
    train, _ = corpus('UD_Italian-ISDT','train')
    dev, _ = corpus('UD_Italian-ISDT','dev')
    modern = [normalize(' '.join(r['words'])) for r in train]
    historical_train = [normalize(' '.join(r['paragraphs'])) for r in stories if r['split']=='train']
    development = [normalize(' '.join(r['words'])) for r in dev]
    development = [t for t in development if 80 <= len(t.replace(' ','')) <= 350 and t not in set(modern)][:12]
    historical_dev = [r for r in stories if r['split']=='dev']
    for work in ('novellino','decameron'):
        selected = [w for r in historical_dev if r['work']==work for w in windows(normalize(' '.join(r['paragraphs'])))]
        development.extend(selected[:12])
    prior = CharacterPrior.fit(modern+historical_train)
    prior.save(STATE/'prior.npz')
    probabilities, counts = language_model(modern+historical_train, spaces=False)
    np.savez(STATE/'legacy-prior.npz',log_probabilities=probabilities,letter_counts=counts)
    write(STATE/'chunks.json',chunk_counts(modern+historical_train))
    lexicon = read(ROOT/'artifacts/segmentation/model.json')['lexicon']
    trials = []
    for weight in (1,4,16):
        model = fit(modern+historical_train*weight,lexicon)
        for alpha in (.1,1.):
            parameters = dict(alpha=alpha,bigram=.5,unknown=3.)
            segmenter = Segmenter(model,**parameters)
            errors = sum(edit_distance(segmenter.segment(t.replace(' ','')).split(),t.split()) for t in development)
            row = dict(weight=weight,parameters=parameters,word_error_rate=errors/sum(len(t.split()) for t in development))
            trials.append(row)
            print('Segmentation development',row,flush=True)
    winner = min(trials,key=lambda r:(r['word_error_rate'],r['weight'],r['parameters']['alpha']))
    write(STATE/'segmenter.json',fit(modern+historical_train*winner['weight'],lexicon))
    passages = []
    for work in ('novellino','decameron'):
        for r in [r for r in historical_dev if r['work']==work][:2]:
            dense = normalize(' '.join(r['paragraphs'])).replace(' ','')[:1400]
            passages.append(dict(id=r['id'],text=dense))
    beam_trials = []
    for width in (128,512,2048,8192):
        rows = []
        for i,p in enumerate(passages):
            rng = random.Random(7123+i)
            alphabet = list(ALPHABET); rng.shuffle(alphabet)
            for family in ('substitution','homophonic'):
                if family=='substitution':
                    cipher = ''.join(dict(zip(ALPHABET,alphabet))[c] for c in p['text']); units = list(cipher); capacity=1
                else:
                    cipher,_ = homophonic(p['text'],rng); units=cipher.split(); capacity=2
                result = beam_search(units,prior,width=width,max_homophones=capacity,cap=60)
                cer = edit_distance(result.get('recovered',''),p['text'])/len(p['text'])
                row = dict(id=p['id'],family=family,cer=cer,status=result['status'],seconds=result['seconds'])
                rows.append(row)
                print('Beam development',width,row,flush=True)
        beam_trials.append(dict(width=width,mean_cer=sum(r['cer'] for r in rows)/len(rows),cases=rows))
    width = min(beam_trials,key=lambda r:(r['mean_cer'],r['width']))['width']
    # Fresh non-Dante candidate sets: compare old mean selection and MDL, without gold in either selector.
    selection = []
    legacy_prior = dict(np.load(STATE/'legacy-prior.npz')); legacy_prior['chunks']=read(STATE/'chunks.json')
    for i,p in enumerate(passages):
        rng = random.Random(9901+i); symbols=list(ALPHABET);rng.shuffle(symbols)
        cipher = ''.join(dict(zip(ALPHABET,symbols))[c] for c in p['text'])
        result = legacy_solve(cipher,legacy_prior,seed=88+i,restarts=4,steps=12000,cap=40)
        scored = score_legacy(cipher,result,prior)
        selected = min(range(len(scored)),key=lambda n:scored[n]['mdl']['total_bits'])
        selection.append(dict(id=p['id'],old_selected=result['selected'],mdl_selected=selected,
            old_cer=edit_distance(result['recovered'],p['text'])/len(p['text']),
            mdl_cer=edit_distance(scored[selected]['recovered'],p['text'])/len(p['text']),
            candidates=[dict(method=r['method'],bits=r['mdl']['total_bits'],cer=edit_distance(r['recovered'],p['text'])/len(p['text'])) for r in scored]))
        print('Selection development',selection[-1],flush=True)
    write(OUT/'development.json',dict(segmentation=trials,selected_segmenter=winner,selected_width=width,
        beam=beam_trials,selection=selection,development_items=len(development),
        training_words=dict(modern=sum(len(t.split()) for t in modern),historical=sum(len(t.split()) for t in historical_train)),
        historical_lexicon_forms=len(set(' '.join(historical_train).split())),
        added_historical_forms=len(set(' '.join(historical_train).split())-set(lexicon))))


def score_legacy(cipher,result,prior):
    candidates=[]
    for row in result['candidates']:
        units=dict(representations(cipher))[row['representation']]
        mapping=row.get('mapping') or infer_mapping(units,row['recovered'])
        mdl=description_length(cipher,row['representation'],mapping,prior)
        candidates.append(dict(row,mapping=mapping,mdl=mdl))
    return candidates


def freeze():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Already frozen')
    check_freeze(); historical()
    inputs = [OUT/'PROTOCOL.md',OUT/'sources.json',OUT/'development.json',OUT/'requirements.txt',OUT/'historical-prose.tar.gz',
              STATE/'prior.npz',STATE/'legacy-prior.npz',STATE/'chunks.json',STATE/'segmenter.json',
              ROOT/'experiments/decipherment-sources.json',ROOT/'experiments/language-sources.json',
              ROOT/'artifacts/segmentation/model.json',ROOT/'artifacts/segmentation/frozen.json']
    write(OUT/'freeze.json',dict(code_sha256=hashes(),inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        development=read(OUT/'development.json'),at=time.time()))


def verify(require_commit=True):
    frozen=read(OUT/'freeze.json')
    if hashes()!=frozen['code_sha256']: raise ValueError('Frozen code drift')
    for p,h in frozen['inputs'].items():
        if digest(ROOT/p)!=h: raise ValueError('Frozen input drift: '+p)
    if require_commit:
        paths=CODE+['experiments/standard-decipherment/freeze.json','experiments/standard-decipherment/PROTOCOL.md',
                    'experiments/standard-decipherment/development.json','experiments/standard-decipherment/sources.json']
        for p in paths:
            committed=subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)
            if hashlib.sha256(committed).hexdigest()!=digest(ROOT/p): raise ValueError('Freeze not committed: '+p)
    return frozen


def prepare():
    frozen=verify()
    if (STATE/'public.json').exists(): raise FileExistsError('Already prepared')
    excluded={'modern':set(),'historical':set()}
    for r in read(ROOT/'artifacts/decipherment/evaluator-only/answers.json'):
        excluded['historical'].update(r['source_sentences'])
    for folder in ('segmentation','codebook-free'):
        for r in read(ROOT/f'artifacts/{folder}/evaluator-only/answers.json'):
            excluded[r['dataset']].update(r['source_ids'])
    seen=set()
    for split in ('train','dev'):
        rows,_=corpus('UD_Italian-ISDT',split)
        seen.update(normalize(' '.join(r['words'])) for r in rows)
    passages=[]
    for dataset,repo,split in [('modern','UD_Italian-ISDT','test'),('historical','UD_Italian-Old','train')]:
        rows,source=corpus(repo,split)
        for region in range(2):
            texts=[]; ids=[]; length=0
            for row in rows[region*len(rows)//2:(region+1)*len(rows)//2]:
                text=normalize(' '.join(row['words'])); n=len(text.replace(' ',''))
                if row['id'] in excluded[dataset] or text in seen or not text or n>1800:
                    texts=[];ids=[];length=0;continue
                if length+n>1800: texts=[];ids=[];length=0
                texts.append(text);ids.append(row['id']);length+=n
                if length>=1200: break
            if length<1200: raise ValueError('Not enough fresh source text')
            excluded[dataset].update(ids)
            passages.append(dict(dataset=dataset,source=source,source_ids=ids,plaintext=' '.join(texts),
                passage=hashlib.sha256((repo+':'.join(ids)).encode()).hexdigest()[:16]))
    vendor_dir=ROOT/'artifacts/decipherment/vendor'
    for f in read(ROOT/'experiments/decipherment-sources.json')['files']:
        if digest(ROOT/f['path'])!=f['sha256']: raise ValueError('Encoder drift')
    spec=importlib.util.spec_from_file_location('naibbe_published',vendor_dir/'naibbe.py')
    vendor=importlib.util.module_from_spec(spec); cwd=os.getcwd()
    try:
        os.chdir(vendor_dir);spec.loader.exec_module(vendor)
    finally: os.chdir(cwd)
    public=[]; answers=[]
    for passage in passages:
        dense=passage['plaintext'].replace(' ','')
        for family in ('substitution','homophonic','variable-homophonic','naibbe'):
            seed=random.SystemRandom().randrange(2**63);rng=random.Random(seed)
            alphabet=list(ALPHABET);rng.shuffle(alphabet);key=dict(zip(ALPHABET,alphabet));inverse={v:k for k,v in key.items()}
            coded=''.join(key[c] for c in dense)
            if family=='substitution': cipher=coded; roundtrip=''.join(inverse[c] for c in cipher)
            elif family in ('homophonic','variable-homophonic'):
                cipher,mapping=(homophonic if family=='homophonic' else artificial)(dense,rng)
                roundtrip=''.join(mapping[t] for t in cipher.split())
            else:
                random.seed(seed); trace=io.StringIO()
                cipher=' '.join(vendor.encrypt_naibbe(coded,vendor.naibbe_tables,vendor.placeholder_to_glyph,use_78=False,pre_plaintext_file=trace))
                roundtrip=''.join(inverse[c] for c in ''.join(trace.getvalue().split()))
            if roundtrip!=dense: raise ValueError('Roundtrip failed')
            ident=hashlib.sha256((cipher+str(seed)).encode()).hexdigest()[:16]
            public.append(dict(id=ident,ciphertext=cipher))
            answers.append(dict(passage,id=ident,family=family,seed=seed,roundtrip=True))
    private=STATE/'evaluator-only';private.mkdir(mode=0o700,exist_ok=True)
    write(STATE/'public.json',sorted(public,key=lambda r:r['id']));write(private/'answers.json',answers)
    write(STATE/'challenge.json',dict(public_sha256=digest(STATE/'public.json'),answers_sha256=digest(private/'answers.json'),
        freeze_sha256=digest(OUT/'freeze.json'),freeze_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        cases=len(public),independent_passages=len(passages),earlier_sentence_overlap=0,training_sentence_overlap=0,at=time.time()))
    print('Prepared 16 cases; references evaluator-only',flush=True)


def solve():
    frozen=verify()
    if (STATE/'predictions.json').exists(): raise FileExistsError('Predictions frozen')
    prior=CharacterPrior.load(STATE/'prior.npz'); legacy=dict(np.load(STATE/'legacy-prior.npz'));legacy['chunks']=read(STATE/'chunks.json')
    config=frozen['development']
    segmenter=Segmenter(read(STATE/'segmenter.json'),**config['selected_segmenter']['parameters'])
    results=[];started=time.time()
    for i,case in enumerate(read(STATE/'public.json')):
        checkpoint=STATE/'partial'/f"{case['id']}.json"
        if checkpoint.exists():
            cached=read(checkpoint)
            if cached['freeze_sha256']!=digest(OUT/'freeze.json') or cached['public_sha256']!=digest(STATE/'public.json'):
                raise ValueError('Checkpoint provenance drift')
            results.append(cached['result']);continue
        cipher=case['ciphertext']; row=legacy_solve(cipher,legacy,seed=4242+i,restarts=4,steps=12000,cap=40)
        candidates=score_legacy(cipher,row,prior); legacy_count=len(candidates)
        old_selected=row['selected']; mdl_legacy=min(range(len(candidates)),key=lambda n:candidates[n]['mdl']['total_bits'])
        runs=[]
        for representation,units in representations(cipher):
            capacity=max(2,math.ceil(len(set(units))/len(ALPHABET))*2)
            for limit in sorted({1,2,capacity}):
                result=beam_search(units,prior,width=config['selected_width'],max_homophones=limit,cap=60)
                runs.append(dict(representation=representation,capacity=limit,**{k:v for k,v in result.items() if k not in ('recovered','mapping')}))
                if result['status']=='complete':
                    mdl=description_length(cipher,representation,result['mapping'],prior)
                    candidates.append(dict(result,representation=representation,method='published-beam',mdl=mdl))
        selected=min(range(len(candidates)),key=lambda n:candidates[n]['mdl']['total_bits'])
        units=cipher.split() if len(cipher.split())>1 else list(cipher)
        em=hmm_em(units,prior,restarts=8,iterations=200,seed=2242+i,cap=60) if len(set(units))<=64 else dict(status='inventory_budget_exceeded')
        result=dict(id=case['id'],candidates=candidates,selected=selected,old_selected=old_selected,
            mdl_legacy_selected=mdl_legacy,legacy_candidates=legacy_count,beam_runs=runs,em=em,
            recovered=candidates[selected]['recovered'],segmented=segmenter.segment(candidates[selected]['recovered']))
        for name,index in [('old_segmented',old_selected),('mdl_legacy_segmented',mdl_legacy)]:
            result[name]=segmenter.segment(candidates[index]['recovered'])
        if 'recovered' in em: em['segmented']=segmenter.segment(em['recovered'])
        write(checkpoint,dict(result=result,freeze_sha256=digest(OUT/'freeze.json'),public_sha256=digest(STATE/'public.json')))
        results.append(result)
        print(f"Ciphertext-only recovery {len(results)}/16 ({time.time()-started:.1f}s this invocation)",flush=True)
    # Segment exact letters from the public monoalphabetic cases is impossible without the key.
    # The segmentation-only control is generated and solved separately before grading.
    write(STATE/'predictions.json',dict(results=results,code_sha256=hashes(),public_sha256=digest(STATE/'public.json'),
        freeze_sha256=digest(OUT/'freeze.json'),seconds_this_invocation=time.time()-started,at=time.time()))


def segment_controls():
    verify()
    if (STATE/'segmentation-predictions.json').exists(): raise FileExistsError('Control predictions frozen')
    if not (STATE/'predictions.json').exists():
        raise ValueError('Freeze cipher predictions before exposing dense-letter controls')
    # This stage only exposes dense letters to segmenters; no boundary information is passed.
    references=read(STATE/'evaluator-only/answers.json')
    cases={r['passage']:r['plaintext'].replace(' ','') for r in references}
    new=Segmenter(read(STATE/'segmenter.json'),**read(OUT/'development.json')['selected_segmenter']['parameters'])
    old=Segmenter(read(ROOT/'artifacts/segmentation/model.json'),**read(ROOT/'artifacts/segmentation/frozen.json')['parameters'])
    rows=[dict(passage=p,new=new.segment(text),old=old.segment(text)) for p,text in cases.items()]
    write(STATE/'segmentation-predictions.json',dict(rows=rows,freeze_sha256=digest(OUT/'freeze.json'),at=time.time()))


def evaluate():
    verify();challenge=read(STATE/'challenge.json');pred=read(STATE/'predictions.json')
    for key,path in [('public_sha256',STATE/'public.json'),('answers_sha256',STATE/'evaluator-only/answers.json'),('freeze_sha256',OUT/'freeze.json')]:
        if challenge[key]!=digest(path): raise ValueError('Challenge drift')
    if pred['code_sha256']!=hashes() or any(pred[k]!=challenge[k] for k in ('public_sha256','freeze_sha256')): raise ValueError('Prediction drift')
    answers={r['id']:r for r in read(STATE/'evaluator-only/answers.json')}
    if {r['id'] for r in pred['results']}!=set(answers) or len(pred['results'])!=len(answers): raise ValueError('Incomplete predictions')
    rows=[]
    for result in pred['results']:
        gold=answers[result['id']]; plain=gold['plaintext']; dense=plain.replace(' ','')
        def metrics(text,segmented):
            ce=edit_distance(text,dense);we=edit_distance(segmented.split(),plain.split())
            return dict(character_errors=ce,word_errors=we,cer=ce/len(dense),wer=we/len(plain.split()),gate=ce/len(dense)<=.01 and we/len(plain.split())<=.1)
        methods={'combined':metrics(result['recovered'],result['segmented']),
                 'legacy_mean':metrics(result['candidates'][result['old_selected']]['recovered'],result['old_segmented']),
                 'legacy_mdl':metrics(result['candidates'][result['mdl_legacy_selected']]['recovered'],result['mdl_legacy_segmented'])}
        if 'recovered' in result['em']: methods['hmm_em']=metrics(result['em']['recovered'],result['em']['segmented'])
        candidate_rows=[dict(method=c['method'],representation=c['representation'],mdl={k:v for k,v in c['mdl'].items() if k!='recovered'},
            cer=edit_distance(c['recovered'],dense)/len(dense)) for c in result['candidates']]
        rows.append(dict(id=result['id'],passage=gold['passage'],dataset=gold['dataset'],family=gold['family'],
            characters=len(dense),words=len(plain.split()),methods=methods,candidates=candidate_rows,
            selected=result['selected'],old_selected=result['old_selected'],mdl_legacy_selected=result['mdl_legacy_selected'],
            oracle_cer=min(c['cer'] for c in candidate_rows),beam_runs=result['beam_runs'],
            em={k:v for k,v in result['em'].items() if k not in ('recovered','segmented')}))
    summary={}
    for family in ('substitution','homophonic','variable-homophonic','naibbe'):
        summary[family]={}
        for dataset in ('modern','historical'):
            subset=[r for r in rows if r['family']==family and r['dataset']==dataset]
            summary[family][dataset]={}
            for method in ('combined','legacy_mean','legacy_mdl','hmm_em'):
                available=[r for r in subset if method in r['methods']]
                if not available: continue
                summary[family][dataset][method]=dict(cases=len(available),cer=sum(r['methods'][method]['character_errors'] for r in available)/sum(r['characters'] for r in available),
                    wer=sum(r['methods'][method]['word_errors'] for r in available)/sum(r['words'] for r in available),gate_passes=sum(r['methods'][method]['gate'] for r in available))
    segment=read(STATE/'segmentation-predictions.json')
    if segment['freeze_sha256']!=digest(OUT/'freeze.json'): raise ValueError('Segmentation provenance drift')
    segments=[]
    for r in segment['rows']:
        gold=next(g for g in answers.values() if g['passage']==r['passage'])
        segments.append(dict(passage=r['passage'],dataset=gold['dataset'],words=len(gold['plaintext'].split()),
            old_errors=edit_distance(r['old'].split(),gold['plaintext'].split()),new_errors=edit_distance(r['new'].split(),gold['plaintext'].split())))
    naibbe=[r for r in rows if r['family']=='naibbe']
    output=dict(summary=summary,cases=rows,segmentation=segments,challenge=challenge,
        predictions_sha256=digest(STATE/'predictions.json'),segmentation_predictions_sha256=digest(STATE/'segmentation-predictions.json'),
        naibbe_gate=all(r['methods']['combined']['gate'] for r in naibbe),voynich_run=False,final_test_scored=False)
    write(OUT/'results.json',output)
    print(json.dumps(dict(summary=summary,segmentation=segments,naibbe_gate=output['naibbe_gate']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['develop','freeze','verify','prepare','solve','segment_controls','evaluate'])
    globals()[parser.parse_args().command]()
