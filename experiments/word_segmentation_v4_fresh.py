"""Freeze word segmentation v4 (mix) and test it fresh: Sacchetti (historical) and unused ParTUT train (modern).

    python -m experiments.word_segmentation_v4_fresh freeze | fetch | check | prepare | solve | evaluate
Perfect letters; references evaluator-only until both methods' predictions are saved. CPU only.
"""
import argparse
import hashlib
import json
import re
import secrets
import subprocess
import urllib.error
import urllib.parse

from bs4 import BeautifulSoup
from experiments.word_segmentation_fresh import fetch as fetch_file, seen_ngrams, passages, extract_book
from experiments.word_segmentation_v2 import ROOT, BASE, read, write, grade
from experiments.word_segmentation_v3 import OUT as V3, verify as v3_frozen
from experiments.word_segmentation_v3_fresh import released_ngrams, solvers
from experiments.word_segmentation_v4 import OUT as V4, CODE as DEV_CODE, historical_types, segmenters
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize
from voynich.unknown_words import Spelling

OUT = ROOT / 'experiments/word-segmentation-v4-fresh'
STATE = ROOT / 'artifacts/word-segmentation-v4-fresh'
RAW = STATE / 'sources'
ROMAN = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
NOVELLE = 40
PARTUT_TRAIN = ROOT / 'artifacts/joint-recovery-v6/sources/it_partut-ud-train.conllu'
RELEASED = ['experiments/word-segmentation-v2/evaluated-records.json', 'experiments/word-segmentation-v3-fresh/evaluated-records.json',
            'experiments/joint-recovery-v5/evaluated-records.json', 'experiments/joint-recovery-v6/evaluated-records.json',
            'experiments/language-id/evaluated-records.json']
FRESH_CODE = ['experiments/word_segmentation_v4_fresh.py']


def roman(n):
    out = ''
    for value, sym in ROMAN:
        while n >= value: out += sym; n -= value
    return out


def freeze():
    if (OUT / 'freeze.json').exists(): raise FileExistsError('Already frozen')
    dev = read(V4 / 'development.json')
    if dev['selected'] != 'mix': raise ValueError('Unexpected selection')
    for p, h in dev['code_sha256'].items():
        if digest(ROOT / p) != h: raise ValueError('Development code drift: ' + p)
    if digest(V4 / 'PROTOCOL.md') != dev['protocol_sha256']: raise ValueError('Protocol drift')
    f3 = v3_frozen(); model = read(BASE)
    lexicon_spelling = Spelling(model['lexicon'], f3['selected']['order']).digest()
    if lexicon_spelling != f3['spelling_sha256']: raise ValueError('v3 spelling drift')
    paths = DEV_CODE + FRESH_CODE + ['experiments/word-segmentation-v4/PROTOCOL.md', 'experiments/word-segmentation-v4/development.json',
                                     'experiments/word-segmentation-v3/freeze.json']
    write(OUT / 'freeze.json', dict(files={p: digest(ROOT / p) for p in paths}, selected='mix', weights=[.5, .5],
          historical_spelling_sha256=dev['historical_spelling_sha256'], lexicon_spelling_sha256=lexicon_spelling,
          fresh_plan=dict(historical='Franco Sacchetti, Il Trecentonovelle, Wikisource, novelle I-XL in numerical order',
                          modern='UD Italian ParTUT train sentences not released by round six',
                          transfer_threshold='historical pooled WER at least 2 points below v3; modern no more than 1 point worse'),
          version=1))
    print('Frozen; commit before fetch')


def verify():
    f = read(OUT / 'freeze.json')
    for p, h in list(f['files'].items()) + [('experiments/word-segmentation-v4-fresh/freeze.json', digest(OUT / 'freeze.json'))]:
        if digest(ROOT / p) != h: raise ValueError('Frozen drift: ' + p)
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)).hexdigest() != h: raise ValueError('Not committed: ' + p)
    return f


def fetch():
    verify()
    if (OUT / 'sources.json').exists(): raise FileExistsError('Sources pinned')
    files, missing = [], []
    for n in range(1, NOVELLE + 1):
        url = 'https://it.wikisource.org/wiki/' + urllib.parse.quote(f'Il_Trecentonovelle/{roman(n)}')
        try:
            r = fetch_file(url, RAW / f'sacchetti-{n:03}.html')
        except urllib.error.HTTPError:
            missing.append(n); continue
        m = re.search(r'"wgRevisionId":(\d+)', (RAW / f'sacchetti-{n:03}.html').read_text())
        r.update(novella=n, revision=int(m[1]) if m else None, author='Franco Sacchetti', work=f'Il Trecentonovelle, {roman(n)}',
                 license='Public-domain original; Wikisource transcription CC BY-SA.')
        files.append(r)
    write(OUT / 'sources.json', dict(files=files, missing_novelle=missing, partut_train=dict(path=str(PARTUT_TRAIN.relative_to(ROOT)), sha256=digest(PARTUT_TRAIN)),
                                     attribution='Franco Sacchetti; Wikisource contributors. ParTUT: Sanguinetti, Bosco and UD contributors (CC BY-NC-SA 4.0).'))
    print(len(files), 'novelle fetched; missing', missing)


def extract_sacchetti(html, n):
    """Body paragraphs: drop each novella's italic argument paragraph, then the v2 extractor's cleaning."""
    soup = BeautifulSoup(html, 'html.parser'); root = soup.select_one('#box_esterno')
    if root is None: raise ValueError('Missing Wikisource text container')
    rubrics = []
    for p in root.select('p'):
        text = ' '.join(p.get_text().split())
        italic = ' '.join(' '.join(i.get_text().split()) for i in p.select('i'))
        if text and italic == text: rubrics.append(text); p.extract()
    rows = extract_book(str(soup))
    return [dict(id=f'sacchetti-{n:03}-p{i + 1:03}', text=r['text']) for i, r in enumerate(rows)], rubrics


def historical_rows():
    rows, rubrics = [], []
    for r in read(OUT / 'sources.json')['files']:
        body, rub = extract_sacchetti((ROOT / r['path']).read_text(), r['novella']); rows += body; rubrics += rub
    return rows, rubrics


def released():
    ids, grams = set(), set()
    for path in RELEASED:
        for r in read(ROOT / path)['answers']:
            ids.update(i.split(':part')[0] for i in (r.get('source_ids') or []))
            w = r['plaintext'].split(); grams.update(tuple(w[i:i + 20]) for i in range(len(w) - 19))
    return ids, grams


def modern_rows():
    ids, _ = released()
    rows = [dict(id='partut-train:' + r['id'], text=normalize(' '.join(r['words']))) for r in conllu_sentences(PARTUT_TRAIN)]
    return [r for r in rows if r['id'] not in ids]


def check():
    if (OUT / 'extraction.json').exists(): raise FileExistsError('Extraction recorded')
    rows, rubrics = historical_rows(); modern = modern_rows()
    write(OUT / 'extraction.json', dict(historical_paragraphs=len(rows), historical_letters=sum(len(r['text'].replace(' ', '')) for r in rows),
          rubrics_removed=len(rubrics), rubric_samples=rubrics[:5], first_paragraph=rows[0]['text'][:200],
          modern_sentences=len(modern), note='Recorded before passage construction; no decoder output read.'))
    print(read(OUT / 'extraction.json'))


def prepare():
    verify()
    if (STATE / 'public.json').exists(): raise FileExistsError('Challenge exists')
    for r in read(OUT / 'sources.json')['files']:
        if digest(ROOT / r['path']) != r['sha256']: raise ValueError('Source drift')
    _, grams = released(); seen = seen_ngrams() | released_ngrams() | grams
    public, answers, excluded = [], [], {}
    for dataset, rows in [('historical', historical_rows()[0]), ('modern', modern_rows())]:
        blocks, rejected = passages(rows, seen); excluded[dataset] = rejected
        for block in blocks:
            ident = secrets.token_hex(12)
            public.append(dict(id=ident, text=block['plaintext'].replace(' ', ''))); answers.append(dict(id=ident, dataset=dataset, **block))
    private = STATE / 'evaluator-only'; private.mkdir(parents=True, exist_ok=True, mode=0o700)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
          freeze_sha256=digest(OUT / 'freeze.json'), sources_sha256=digest(OUT / 'sources.json'), extraction_sha256=digest(OUT / 'extraction.json'),
          head_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), cases=len(public), excluded_source_ids=excluded))
    print('Prepared', len(public), 'cases')


def solve():
    verify(); c = read(STATE / 'challenge.json')
    if (STATE / 'predictions.json').exists(): raise FileExistsError('Predictions frozen')
    model = read(BASE); f3 = v3_frozen()
    v3 = solvers(f3)['v3']; methods, hist = segmenters(model, f3)
    if hist.digest() != read(OUT / 'freeze.json')['historical_spelling_sha256']: raise ValueError('Historical spelling drift')
    rows = [dict(id=case['id'], v3=v3.segment(case['text']), v4=methods['mix'].segment(case['text'])) for case in read(STATE / 'public.json')]
    write(STATE / 'predictions.json', dict(rows=rows, public_sha256=c['public_sha256'], freeze_sha256=c['freeze_sha256']))
    print('Solved', len(rows))


def evaluate():
    verify(); c = read(STATE / 'challenge.json'); pred = read(STATE / 'predictions.json')
    if digest(STATE / 'evaluator-only/answers.json') != c['answers_sha256'] or pred['public_sha256'] != c['public_sha256']: raise ValueError('Drift')
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    rows = [dict(id=r['id'], dataset=answers[r['id']]['dataset'], **{m: grade(r[m], answers[r['id']]['plaintext']) for m in ('v3', 'v4')}) for r in pred['rows']]
    def pool(sub, m):
        s = {k: sum(x[m][k] for x in sub) for k in ('words', 'errors', 'fp', 'fn')}
        return dict(**s, wer=s['errors'] / s['words'], gate_passes=sum(x[m]['wer'] <= .1 for x in sub))
    summary = {d: {m: pool([x for x in rows if x['dataset'] == d], m) for m in ('v3', 'v4')} for d in ('historical', 'modern')}
    transfer = (summary['historical']['v3']['wer'] - summary['historical']['v4']['wer'] >= .02
                and summary['modern']['v4']['wer'] - summary['modern']['v3']['wer'] <= .01)
    write(OUT / 'results.json', dict(summary=summary, cases=rows, transfer_passed=transfer, challenge=c,
                                     predictions_sha256=digest(STATE / 'predictions.json'), voynich_used=False))
    write(OUT / 'evaluated-records.json', dict(public=read(STATE / 'public.json'), answers=list(answers.values()), predictions=pred))
    print(json.dumps({d: {m: (round(100 * v['wer'], 2), v['fp'], v['fn'], v['gate_passes']) for m, v in s.items()} for d, s in summary.items()}), 'transfer', transfer)
    for x in rows: print(x['dataset'], round(100 * x['v3']['wer'], 2), '->', round(100 * x['v4']['wer'], 2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['freeze', 'fetch', 'check', 'prepare', 'solve', 'evaluate'])
    globals()[p.parse_args().command]()
