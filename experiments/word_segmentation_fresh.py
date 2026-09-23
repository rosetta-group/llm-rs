"""Prepare, solve and grade fresh perfect-letter word segmentation after a Git freeze.

The solver receives dense letters only; references are evaluator-only until predictions
are saved. Corpus extraction never inspects model scores. CPU, no paid services.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup
from experiments.word_segmentation_v2 import ROOT, OUT, STATE, BASE, read, write, verify, grade
from experiments.historical_sources import verify as historical
from experiments.verse_sources import verify as verse
from experiments.segmentation import corpus
from voynich.corpora import conllu_sentences
from voynich.decipher import normalize
from voynich.data import digest
from voynich.segmentation import Segmenter

HEADERS = {'User-Agent':'VoynichMethodsBenchmark/word-segmentation-v2 (public research)'}
MINIMUM, MAXIMUM, CASES = 5200, 6000, 4


def fetch(url, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        request=urllib.request.Request(url,headers=HEADERS)
        path.write_bytes(urllib.request.urlopen(request,timeout=60).read())
    return dict(url=url,path=str(path.relative_to(ROOT)),sha256=digest(path))


def extract_book(html):
    soup=BeautifulSoup(html,'html.parser');root=soup.select_one('#box_esterno')
    if root is None: raise ValueError('Missing Wikisource text container')
    for node in root.select('script,style,table,sup,.ws-noexport,.noprint,.references,.mw-editsection,.intestazione,.intest-normal,.barraCapitolo,.box_sottopagina,.AltraVersione'):
        node.decompose()
    for br in root.select('br'): br.replace_with(' ')
    rows=[];chapter='unlabelled'
    for node in root.select('h2,h3,h4,p'):
        if node.name!='p':
            chapter=' '.join(node.get_text(' ').split());continue
        text=normalize(' '.join(node.get_text().split()))
        if len(text.split())>=8:
            rows.append(dict(id=f'villani-book1-p{len(rows)+1:04}',chapter=chapter,text=text))
    if not rows: raise ValueError('No prose extracted')
    return rows


def seen_ngrams(n=20):
    texts=[]
    for split in ('train','dev'):
        rows,_=corpus('UD_Italian-ISDT',split)
        texts.extend(normalize(' '.join(r['words'])) for r in rows)
    texts.extend(normalize(' '.join(r['paragraphs'])) for r in historical())
    texts.extend(normalize(' '.join(r['lines'])) for r in verse())
    seen=set()
    for text in texts:
        words=text.split();seen.update(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    return seen


def passages(rows,seen):
    built=[];parts=[];ids=[];size=0;rejected=[]
    for row in rows:
        words=row['text'].split()
        if any(tuple(words[i:i+20]) in seen for i in range(len(words)-19)):
            rejected.append(row['id']);continue
        # Overlong paragraphs are split at word boundaries; normal sentences/paragraphs remain intact.
        chunks=[];chunk=[];length=0
        for w in words:
            if len(w)>MAXIMUM: raise ValueError('A word exceeds the passage maximum')
            if length+len(w)>MAXIMUM:
                chunks.append(chunk);chunk=[];length=0
            chunk.append(w);length+=len(w)
        if chunk:chunks.append(chunk)
        for j,words in enumerate(chunks):
            length=sum(map(len,words));ident=row['id']+f':part{j}'
            if size+length>MAXIMUM:
                # The previous residual is shorter than MINIMUM. Keep its source IDs in the
                # excluded residual list instead of silently treating it as an evaluated passage.
                rejected.extend(ids);parts=[];ids=[];size=0
            parts.extend(words);ids.append(ident);size+=length
            if size>=MINIMUM:
                built.append(dict(plaintext=' '.join(parts),source_ids=ids,characters=size))
                parts=[];ids=[];size=0
                if len(built)==CASES:return built,rejected
    raise ValueError(f'Only {len(built)}/{CASES} fresh passages; no corpus substitution allowed')


def prepare():
    frozen=verify()
    if (STATE/'public.json').exists():raise FileExistsError('Challenge already exists')
    raw=STATE/'sources';files=[]
    book_url='https://it.wikisource.org/wiki/Nuova_Cronica/Libro_primo'
    record=fetch(book_url,raw/'villani-book1.html');files.append(record)
    html=(raw/'villani-book1.html').read_text()
    match=re.search(r'"wgRevisionId":(\d+)',html)
    if not match:raise ValueError('Missing Wikisource revision ID')
    record.update(revision=int(match[1]),permanent_url='https://it.wikisource.org/w/index.php?oldid='+match[1],
                  author='Giovanni Villani',work='Nuova Cronica, Libro primo',date='14th century',
                  license='Public-domain original; Wikisource transcription CC BY-SA. Retain source/history attribution.')
    # Immutable GitHub commit for the fresh modern test and its attribution/license files.
    api='https://api.github.com/repos/UniversalDependencies/UD_Italian-VIT/commits/master'
    files.append(fetch(api,raw/'vit-commit.json'));revision=read(raw/'vit-commit.json')['sha']
    for name in ('it_vit-ud-test.conllu','README.md','LICENSE.txt'):
        r=fetch(f'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-VIT/{revision}/{name}',raw/name)
        r.update(revision=revision,repository='UD_Italian-VIT',license='CC BY-NC-SA 3.0');files.append(r)
    modern=[dict(id='vit:'+r['id'],text=normalize(' '.join(r['words']))) for r in conllu_sentences(raw/'it_vit-ud-test.conllu')]
    historical_rows=extract_book(html)
    sources=dict(files=files,historical_paragraphs=len(historical_rows),modern_sentences=len(modern),
                 historical_extraction='Book I paragraphs in page order, headings/nav/notes dropped; edited transcription, not diplomatic.',
                 author_attribution='Giovanni Villani; Wikisource contributors. VIT: Fabio Tamburini, Maria Simi, Cristina Bosco and UD contributors.')
    write(OUT/'sources.json',sources)
    seen=seen_ngrams();public=[];answers=[];excluded={}
    for dataset,rows in [('historical',historical_rows),('modern',modern)]:
        blocks,rejected=passages(rows,seen);excluded[dataset]=rejected
        for block in blocks:
            ident=secrets.token_hex(12);dense=block['plaintext'].replace(' ','')
            public.append(dict(id=ident,text=dense))
            answers.append(dict(id=ident,dataset=dataset,**block))
    private=STATE/'evaluator-only';private.mkdir(parents=True,exist_ok=True,mode=0o700)
    write(STATE/'public.json',sorted(public,key=lambda r:r['id']));write(private/'answers.json',answers)
    write(STATE/'challenge.json',dict(public_sha256=digest(STATE/'public.json'),answers_sha256=digest(private/'answers.json'),
          freeze_sha256=digest(OUT/'freeze.json'),sources_sha256=digest(OUT/'sources.json'),
          freeze_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          cases=len(public),excluded_source_ids=excluded,overlap_rule='exact shared 20-word sequence'))
    print('Prepared 8 cases; references remain evaluator-only',flush=True)


def check_challenge():
    verify();c=read(STATE/'challenge.json')
    for key,path in [('public_sha256',STATE/'public.json'),('freeze_sha256',OUT/'freeze.json'),('sources_sha256',OUT/'sources.json')]:
        if digest(path)!=c[key]:raise ValueError('Challenge drift: '+key)
    for r in read(OUT/'sources.json')['files']:
        if digest(ROOT/r['path'])!=r['sha256']:raise ValueError('Source drift')
    return c


def solve():
    f=verify();c=check_challenge()
    if (STATE/'predictions.json').exists():raise FileExistsError('Predictions frozen')
    models=dict(baseline=read(BASE),verse=json.loads(gzip.decompress((OUT/'model.json.gz').read_bytes())))
    solvers={k:Segmenter(m,**f['parameters']) for k,m in models.items()}
    predictions=[]
    for case in read(STATE/'public.json'):
        predictions.append(dict(id=case['id'],**{k:s.segment(case['text']) for k,s in solvers.items()}))
        print(f"Solved {len(predictions)}/{c['cases']}",flush=True)
    write(STATE/'predictions.json',dict(rows=predictions,public_sha256=c['public_sha256'],freeze_sha256=c['freeze_sha256']))


def evaluate():
    c=check_challenge();pred=read(STATE/'predictions.json')
    if digest(STATE/'evaluator-only/answers.json')!=c['answers_sha256']:raise ValueError('Reference drift')
    if any(pred[k]!=c[k] for k in ('public_sha256','freeze_sha256')):raise ValueError('Prediction drift')
    answers={r['id']:r for r in read(STATE/'evaluator-only/answers.json')}
    if len(pred['rows'])!=len(answers) or {r['id'] for r in pred['rows']}!=set(answers):raise ValueError('Incomplete predictions')
    rows=[]
    for r in pred['rows']:
        gold=answers[r['id']]
        rows.append(dict(id=r['id'],dataset=gold['dataset'],source_ids=gold['source_ids'],
                         **{method:grade(r[method],gold['plaintext']) for method in ('baseline','verse')}))
    summary={}
    for dataset in ('historical','modern'):
        sub=[r for r in rows if r['dataset']==dataset];summary[dataset]={}
        for method in ('baseline','verse'):
            stats={k:sum(r[method][k] for r in sub) for k in ('words','errors','tp','fp','fn')}
            summary[dataset][method]=dict(**stats,wer=stats['errors']/stats['words'],
                precision=stats['tp']/max(1,stats['tp']+stats['fp']),recall=stats['tp']/max(1,stats['tp']+stats['fn']),
                f1=2*stats['tp']/max(1,2*stats['tp']+stats['fp']+stats['fn']),
                gate_passes=sum(r[method]['wer']<=.1 for r in sub))
    transfer=(summary['historical']['baseline']['wer']-summary['historical']['verse']['wer']>=.03
              and summary['modern']['verse']['wer']-summary['modern']['baseline']['wer']<=.01)
    result=dict(summary=summary,cases=rows,transfer_passed=transfer,word_gate_passed=all(r['verse']['wer']<=.1 for r in rows),
                challenge=c,predictions_sha256=digest(STATE/'predictions.json'),voynich_used=False,naibbe_gate_opened=False)
    write(OUT/'results.json',result)
    # Graded release. Never publish answers before predictions exist.
    write(OUT/'evaluated-records.json',dict(public=read(STATE/'public.json'),answers=list(answers.values()),predictions=pred))
    print(json.dumps(dict(summary=summary,transfer_passed=transfer,word_gate_passed=result['word_gate_passed']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','solve','evaluate']);globals()[p.parse_args().command]()
