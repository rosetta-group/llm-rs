"""Fresh perfect-letter test of the frozen v3 segmenter: Compagni (historical), ParTUT (modern).

Stages: fetch -> check (extraction record, before any passage) -> prepare -> solve -> evaluate.
The solver sees dense letters and opaque IDs only; references stay evaluator-only until
both methods' predictions are saved. No stage reads decoder scores to choose text. CPU only.
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
import re
import secrets
import subprocess
import urllib.parse

from bs4 import BeautifulSoup
from experiments.word_segmentation_v2 import ROOT, BASE, read, write, grade
from experiments.word_segmentation_fresh import fetch, seen_ngrams, passages
from experiments.word_segmentation_v3 import OUT as V3, V2, verify as frozen, is_rubric
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize
from voynich.segmentation import Segmenter
from voynich.unknown_words import Spelling, SpellingSegmenter

OUT = ROOT / 'experiments/word-segmentation-v3-fresh'
STATE = ROOT / 'artifacts/word-segmentation-v3-fresh'
RAW = STATE / 'sources'
WORK = "Cronica delle cose occorrenti ne' tempi suoi/Cronaca di Dino Compagni/Libro "
BOOKS = ('I', 'II', 'III')
PARTUT_SPLITS = ('test', 'dev')  # test alone has ~3,640 words, short of four 5,200-letter passages
RELEASED = ['experiments/word-segmentation-v2/evaluated-records.json']


def fetch_sources():
    if (OUT / 'sources.json').exists(): raise FileExistsError('Sources already pinned')
    frozen()
    files = []
    for book in BOOKS:
        url = 'https://it.wikisource.org/wiki/' + urllib.parse.quote((WORK + book).replace(' ', '_'))
        record = fetch(url, RAW / f'compagni-libro-{book}.html')
        match = re.search(r'"wgRevisionId":(\d+)', (RAW / f'compagni-libro-{book}.html').read_text())
        if not match: raise ValueError('Missing Wikisource revision ID')
        record.update(revision=int(match[1]), author='Dino Compagni', work='Cronica, Libro ' + book,
                      date='c. 1310-1312', license='Public-domain original; Wikisource transcription CC BY-SA. '
                      'Page revision does not pin transcluded Pagina pages; the archived HTML hash does.')
        files.append(record)
    api = 'https://api.github.com/repos/UniversalDependencies/UD_Italian-ParTUT/commits/master'
    files.append(fetch(api, RAW / 'partut-commit.json'))
    revision = read(RAW / 'partut-commit.json')['sha']
    for name in [f'it_partut-ud-{s}.conllu' for s in PARTUT_SPLITS] + ['README.md', 'LICENSE.txt']:
        r = fetch(f'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParTUT/{revision}/{name}', RAW / name)
        r.update(revision=revision, repository='UD_Italian-ParTUT', license='CC BY-NC-SA 4.0'); files.append(r)
    write(OUT / 'sources.json', dict(files=files, author_attribution='Dino Compagni; Wikisource contributors. '
          'ParTUT: Manuela Sanguinetti, Cristina Bosco and UD contributors.'))
    return files


def extract_compagni(html, book):
    """Body paragraphs of one book; drop notes, navigation and chapter rubrics."""
    soup = BeautifulSoup(html, 'html.parser')
    root = soup.select_one('#box_esterno') or soup.select_one('.mw-parser-output')
    if root is None: raise ValueError('Missing Wikisource text container')
    for node in root.select('script,style,table,sup,.ws-noexport,.noprint,.references,.mw-editsection,'
                            '.intestazione,.intest-normal,.barraCapitolo,.box_sottopagina,.AltraVersione,'
                            '.pagenum,.ws-pagenum,h1,h2,h3,h4'):
        node.decompose()
    rows, rubrics = [], []
    for p in root.select('p'):
        raw = p.get_text()
        if is_rubric(raw) or ROMAN_ONLY.match(raw.strip()):
            rubrics.append(raw); continue
        text = normalize(' '.join(raw.split()))
        if len(text.split()) >= 8:
            rows.append(dict(id=f'compagni-{book}-p{len(rows) + 1:04}', text=text))
    return rows, rubrics


ROMAN_ONLY = re.compile(r'^[IVXLCDM]+\.?$')


def historical_rows():
    rows, rubrics = [], []
    for book in BOOKS:
        r, h = extract_compagni((RAW / f'compagni-libro-{book}.html').read_text(), book)
        rows += r; rubrics += h
    return rows, rubrics


def modern_rows():
    return [dict(id=f'partut-{s}:' + r['id'], text=normalize(' '.join(r['words'])))
            for s in PARTUT_SPLITS for r in conllu_sentences(RAW / f'it_partut-ud-{s}.conllu')]


def released_ngrams(n=20):
    """Also reject overlap with every previously released fresh passage."""
    seen = set()
    for path in RELEASED:
        for row in read(ROOT / path)['answers']:
            words = row['plaintext'].split()
            seen.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return seen


def check():
    """Record extraction counts before any passage exists. Reads no decoder output."""
    if (OUT / 'extraction.json').exists(): raise FileExistsError('Extraction recorded')
    rows, rubrics = historical_rows(); modern = modern_rows()
    lengths = sorted(len(r['text'].split()) for r in rows)
    write(OUT / 'extraction.json', dict(
        historical_paragraphs=len(rows), historical_letters=sum(len(r['text'].replace(' ', '')) for r in rows),
        rubrics_removed=len(rubrics), rubric_samples=[' '.join(t.split())[:120] for t in rubrics[:5]],
        paragraph_words_min=lengths[0], paragraph_words_median=lengths[len(lengths) // 2],
        first_paragraph=rows[0]['text'][:200], last_paragraph=rows[-1]['text'][-200:],
        modern_sentences=len(modern), modern_letters=sum(len(r['text'].replace(' ', '')) for r in modern),
        partut_splits=list(PARTUT_SPLITS), note='Recorded before passage construction; no decoder output read.'))
    return read(OUT / 'extraction.json')


def prepare():
    f = frozen()
    if (STATE / 'public.json').exists(): raise FileExistsError('Challenge already exists')
    for r in read(OUT / 'sources.json')['files']:
        if digest(ROOT / r['path']) != r['sha256']: raise ValueError('Source drift')
    extraction = read(OUT / 'extraction.json')
    seen = seen_ngrams() | released_ngrams()
    public, answers, excluded = [], [], {}
    for dataset, rows in [('historical', historical_rows()[0]), ('modern', modern_rows())]:
        blocks, rejected = passages(rows, seen); excluded[dataset] = rejected
        for block in blocks:
            ident = secrets.token_hex(12)
            public.append(dict(id=ident, text=block['plaintext'].replace(' ', '')))
            answers.append(dict(id=ident, dataset=dataset, **block))
    private = STATE / 'evaluator-only'; private.mkdir(parents=True, exist_ok=True, mode=0o700)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
          freeze_sha256=digest(V3 / 'freeze.json'), sources_sha256=digest(OUT / 'sources.json'),
          extraction_sha256=digest(OUT / 'extraction.json'),
          head_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          cases=len(public), excluded_source_ids=excluded, overlap_rule='exact shared 20-word sequence, incl. released passages',
          historical_paragraphs=extraction['historical_paragraphs']))
    print('Prepared', len(public), 'cases; references remain evaluator-only', flush=True)


def check_challenge():
    frozen(); c = read(STATE / 'challenge.json')
    for key, path in [('public_sha256', STATE / 'public.json'), ('freeze_sha256', V3 / 'freeze.json'),
                      ('sources_sha256', OUT / 'sources.json'), ('extraction_sha256', OUT / 'extraction.json')]:
        if digest(path) != c[key]: raise ValueError('Challenge drift: ' + key)
    return c


def solvers(f):
    model = read(BASE)
    spelling = Spelling(model['lexicon'], f['selected']['order'])
    if spelling.digest() != f['spelling_sha256']: raise ValueError('Spelling drift')
    return dict(baseline=Segmenter(model, **f['parameters']),
                v3=SpellingSegmenter(model, spelling, f['unknown_rate'], f['selected']['elision'], **f['parameters']))


def solve():
    f = frozen(); c = check_challenge()
    if (STATE / 'predictions.json').exists(): raise FileExistsError('Predictions frozen')
    methods = solvers(f); rows = []
    for case in read(STATE / 'public.json'):
        rows.append(dict(id=case['id'], **{k: s.segment(case['text']) for k, s in methods.items()}))
        print(f"Solved {len(rows)}/{c['cases']}", flush=True)
    write(STATE / 'predictions.json', dict(rows=rows, public_sha256=c['public_sha256'], freeze_sha256=c['freeze_sha256']))


def summarise(rows, method):
    s = {k: sum(r[method][k] for r in rows) for k in ('words', 'errors', 'tp', 'fp', 'fn')}
    return dict(**s, wer=s['errors'] / s['words'], precision=s['tp'] / max(1, s['tp'] + s['fp']),
                recall=s['tp'] / max(1, s['tp'] + s['fn']), f1=2 * s['tp'] / max(1, 2 * s['tp'] + s['fp'] + s['fn']),
                gate_passes=sum(r[method]['wer'] <= .1 for r in rows))


def evaluate():
    c = check_challenge(); pred = read(STATE / 'predictions.json')
    if digest(STATE / 'evaluator-only/answers.json') != c['answers_sha256']: raise ValueError('Reference drift')
    if any(pred[k] != c[k] for k in ('public_sha256', 'freeze_sha256')): raise ValueError('Prediction drift')
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    if {r['id'] for r in pred['rows']} != set(answers): raise ValueError('Incomplete predictions')
    rows = [dict(id=r['id'], dataset=answers[r['id']]['dataset'], source_ids=answers[r['id']]['source_ids'],
                 **{m: grade(r[m], answers[r['id']]['plaintext']) for m in ('baseline', 'v3')}) for r in pred['rows']]
    summary = {d: {m: summarise([r for r in rows if r['dataset'] == d], m) for m in ('baseline', 'v3')}
               for d in ('historical', 'modern')}
    transfer = (summary['historical']['baseline']['wer'] - summary['historical']['v3']['wer'] >= .03
                and summary['modern']['v3']['wer'] - summary['modern']['baseline']['wer'] <= .01)
    result = dict(summary=summary, cases=rows, transfer_passed=transfer,
                  word_gate_passed=all(r['v3']['wer'] <= .1 for r in rows), challenge=c,
                  predictions_sha256=digest(STATE / 'predictions.json'), voynich_used=False, naibbe_gate_opened=False)
    write(OUT / 'results.json', result)
    write(OUT / 'evaluated-records.json', dict(public=read(STATE / 'public.json'), answers=list(answers.values()), predictions=pred))
    print(json.dumps(dict(summary={d: {m: round(v['wer'], 4) for m, v in s.items()} for d, s in summary.items()},
                          transfer_passed=transfer, word_gate_passed=result['word_gate_passed'])))


def state_files():
    return sorted(RAW.glob('*')) + [STATE / p for p in ('public.json', 'challenge.json', 'predictions.json', 'evaluator-only/answers.json')]


def archive():
    target = OUT / 'fresh-sources.tar.gz'
    if target.exists(): raise FileExistsError('Archive already exists')
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in state_files():
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''; info.mode = 0o644
                with path.open('rb') as handle: tar.addfile(info, handle)
    write(OUT / 'archive.json', dict(archive_sha256=digest(target), files={str(p.relative_to(ROOT)): digest(p) for p in state_files()}))


def restore():
    manifest = read(OUT / 'archive.json'); target = OUT / 'fresh-sources.tar.gz'
    if digest(target) != manifest['archive_sha256']: raise ValueError('Archive drift')
    with tarfile.open(target) as tar:
        if set(tar.getnames()) != set(manifest['files']): raise ValueError('Unexpected archive members')
        for member in tar:
            path = (ROOT / member.name).resolve()
            if not member.isfile() or not path.is_relative_to(STATE.resolve()): raise ValueError('Unsafe member')
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['files'][member.name]: raise ValueError('Member drift')
            if path.exists() and path.read_bytes() != data: raise ValueError('Refusing to replace changed data: ' + member.name)
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)


def verify():
    """Check archive, sources, sealed challenge and freeze commit order; regrade every case."""
    f = frozen(); manifest = read(OUT / 'archive.json')
    if digest(OUT / 'fresh-sources.tar.gz') != manifest['archive_sha256']: raise ValueError('Archive drift')
    for path, expected in manifest['files'].items():
        if digest(ROOT / path) != expected: raise ValueError('Working data drift: ' + path)
    for r in read(OUT / 'sources.json')['files']:
        if digest(ROOT / r['path']) != r['sha256']: raise ValueError('Source drift')
    c = check_challenge(); result = read(OUT / 'results.json')
    if result['challenge'] != c or digest(STATE / 'predictions.json') != result['predictions_sha256']: raise ValueError('Result drift')
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', '85531e7', c['head_commit']], cwd=ROOT)
    if subprocess.check_output(['git', 'show', c['head_commit'] + ':experiments/word-segmentation-v3-fresh/sources.json'], cwd=ROOT) != (OUT / 'sources.json').read_bytes():
        raise ValueError('Sources not committed before passage construction')
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    predictions = {r['id']: r for r in read(STATE / 'predictions.json')['rows']}
    for case in result['cases']:
        for m in ('baseline', 'v3'):
            if grade(predictions[case['id']][m], answers[case['id']]['plaintext']) != case[m]: raise ValueError('Grade drift')
    return dict(verified=True, cases=len(result['cases']), transfer_passed=result['transfer_passed'],
                word_gate_passed=result['word_gate_passed'], selected=f['selected'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['fetch', 'check', 'prepare', 'solve', 'evaluate', 'archive', 'restore', 'verify'])
    command = p.parse_args().command
    print({'fetch': fetch_sources, 'check': check}.get(command, lambda: globals()[command]())())
