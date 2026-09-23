"""Archive and regrade the completed word-segmentation comparison without tuning."""
import argparse
import gzip
import json
import re
import subprocess
import tarfile

from bs4 import BeautifulSoup
from experiments.word_segmentation_v2 import ROOT, OUT, STATE, BASE, read, write, verify as freeze, grade, choose
from experiments.word_segmentation_fresh import extract_book, passages, seen_ngrams
from experiments.verse_sources import verify as verse_sources
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize
from voynich.verse_word_model import augment


def archive():
    target = OUT / 'fresh-sources.tar.gz'
    if target.exists():
        raise FileExistsError('Archive already exists')
    files = sorted((STATE / 'sources').glob('*')) + [STATE / p for p in (
        'public.json', 'evaluator-only/answers.json', 'challenge.json', 'predictions.json')]
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in sorted(files):
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(ROOT)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as handle:
                    tar.addfile(info, handle)
    write(OUT / 'archive.json', dict(archive_sha256=digest(target),
          files={str(p.relative_to(ROOT)): digest(p) for p in files},
          attribution='See sources.json and the archived VIT README.md and LICENSE.txt.'))


def restore():
    manifest = read(OUT / 'archive.json')
    target = OUT / 'fresh-sources.tar.gz'
    if digest(target) != manifest['archive_sha256']:
        raise ValueError('Archive drift')
    import hashlib
    with tarfile.open(target) as tar:
        if set(tar.getnames()) != set(manifest['files']):
            raise ValueError('Unexpected archive members')
        for member in tar:
            path = (ROOT / member.name).resolve()
            if not member.isfile() or not path.is_relative_to(STATE.resolve()):
                raise ValueError('Unsafe member')
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['files'][member.name]:
                raise ValueError('Member drift')
            if path.exists() and path.read_bytes() != data:
                raise ValueError('Refusing to replace changed working data: ' + member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def source_audit():
    """Post-grading extraction audit; never change the released challenge or scores."""
    html = (STATE / 'sources/villani-book1.html').read_text()
    extracted = extract_book(html)
    root = BeautifulSoup(html, 'html.parser').select_one('#box_esterno')
    # Actual source encodes chapter rubrics as ordinary paragraphs, not h2/h3/h4.
    rubrics = {normalize(p.get_text()) for p in root.select('p')
               if re.match(r'^\s*[IVXLCDM]+\s*\n', p.get_text())}
    heading_ids = {r['id'] for r in extracted if r['text'] in rubrics}
    sources = {r['id']: r['text'] for r in extracted}
    answers = read(OUT / 'evaluated-records.json')['answers']
    cases = []
    for row in answers:
        if row['dataset'] != 'historical':
            continue
        headings = [ident for ident in row['source_ids'] if ident.split(':part')[0] in heading_ids]
        words = sum(len(sources[ident.split(':part')[0]].split()) for ident in headings)
        cases.append(dict(id=row['id'],heading_source_ids=headings,heading_words=words))
    return dict(post_grading=True, headings_included=True,
                extracted_paragraphs=len(extracted),heading_paragraphs=len(heading_ids),cases=cases,
                evaluated_heading_words=sum(r['heading_words'] for r in cases),
                note='Protocol deviation: ordinary-p chapter rubrics survived. No corrected challenge or selective regrade.')


def verify():
    freeze()
    manifest = read(OUT / 'archive.json')
    if digest(OUT / 'fresh-sources.tar.gz') != manifest['archive_sha256']:
        raise ValueError('Archive drift')
    for path, expected in manifest['files'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Working data drift: ' + path)
    result = read(OUT / 'results.json'); release = read(OUT / 'evaluated-records.json')
    challenge = read(STATE / 'challenge.json')
    if result['challenge'] != challenge:
        raise ValueError('Challenge drift')
    checks = [('public_sha256', STATE / 'public.json'),
              ('answers_sha256', STATE / 'evaluator-only/answers.json'),
              ('freeze_sha256', OUT / 'freeze.json'), ('sources_sha256', OUT / 'sources.json')]
    for key, path in checks:
        if digest(path) != challenge[key]:
            raise ValueError('Challenge hash mismatch: ' + key)
    committed = subprocess.check_output(['git', 'show', challenge['freeze_commit'] +
        ':experiments/word-segmentation-v2/freeze.json'], cwd=ROOT)
    if committed != (OUT / 'freeze.json').read_bytes():
        raise ValueError('Pre-evaluation freeze mismatch')
    for row in read(OUT / 'sources.json')['files']:
        if digest(ROOT / row['path']) != row['sha256']:
            raise ValueError('Source drift')
    if release != dict(public=read(STATE / 'public.json'),
                       answers=read(STATE / 'evaluator-only/answers.json'),
                       predictions=read(STATE / 'predictions.json')):
        raise ValueError('Release mismatch')
    if digest(STATE / 'predictions.json') != result['predictions_sha256']:
        raise ValueError('Predictions changed')
    answers = {r['id']: r for r in release['answers']}
    public = {r['id']: r['text'] for r in release['public']}
    predictions = {r['id']: r for r in release['predictions']['rows']}
    if len(answers) != 8 or set(answers) != set(public) or set(answers) != set(predictions):
        raise ValueError('Case IDs mismatch')
    for case in result['cases']:
        ref = answers[case['id']]
        if ref['plaintext'].replace(' ', '') != public[case['id']]:
            raise ValueError('Dense text mismatch')
        if case['dataset'] != ref['dataset'] or case['source_ids'] != ref['source_ids']:
            raise ValueError('Case metadata mismatch')
        for method in ('baseline', 'verse'):
            if grade(predictions[case['id']][method], ref['plaintext']) != case[method]:
                raise ValueError('Grade drift')
    # Reconstruct passage selection from archived sources and the original overlap rule.
    historical = extract_book((STATE / 'sources/villani-book1.html').read_text())
    modern = [dict(id='vit:' + r['id'], text=normalize(' '.join(r['words'])))
              for r in conllu_sentences(STATE / 'sources/it_vit-ud-test.conllu')]
    seen = seen_ngrams()
    for dataset, rows in [('historical', historical), ('modern', modern)]:
        blocks, excluded = passages(rows, seen)
        expected = [{k: r[k] for k in ('plaintext', 'source_ids', 'characters')}
                    for r in release['answers'] if r['dataset'] == dataset]
        if blocks != expected or excluded != challenge['excluded_source_ids'][dataset]:
            raise ValueError('Passage construction drift')
        for method, reported in result['summary'][dataset].items():
            sub = [r[method] for r in result['cases'] if r['dataset'] == dataset]
            sums = {k: sum(r[k] for r in sub) for k in ('words', 'errors', 'tp', 'fp', 'fn')}
            actual = dict(**sums, wer=sums['errors']/sums['words'],
                precision=sums['tp']/(sums['tp']+sums['fp']), recall=sums['tp']/(sums['tp']+sums['fn']),
                f1=2*sums['tp']/(2*sums['tp']+sums['fp']+sums['fn']),
                gate_passes=sum(r['wer'] <= .1 for r in sub))
            if actual != reported:
                raise ValueError('Summary drift')
    s = result['summary']
    transfer = s['historical']['baseline']['wer']-s['historical']['verse']['wer'] >= .03 and s['modern']['verse']['wer']-s['modern']['baseline']['wer'] <= .01
    word_gate = all(r['verse']['wer'] <= .1 for r in result['cases'])
    if (result['transfer_passed'], result['word_gate_passed'], result['voynich_used'], result['naibbe_gate_opened']) != (transfer, word_gate, False, False):
        raise ValueError('Decision drift')
    dev = read(OUT / 'development.json')
    if choose(dev['rows']) != dev['selected_weight']:
        raise ValueError('Development selection drift')
    model, train_ids = augment(read(BASE), verse_sources(), dev['selected_weight'])
    if model != json.loads(gzip.decompress((OUT / 'model.json.gz').read_bytes())) or train_ids != dev['training_poem_ids']:
        raise ValueError('Training-only fitted model drift')
    for row in dev['rows']:
        for source, ref in dev['references'].items():
            if grade(row['predictions'][source], ref) != row['grades'][source]:
                raise ValueError('Development grade drift')
    if source_audit() != read(OUT / 'source-audit.json'):
        raise ValueError('Source audit drift')
    return dict(verified=True, fresh_cases=8, development_candidates=4,
                transfer_passed=transfer, word_gate_passed=word_gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['archive', 'restore', 'verify', 'audit'])
    command = parser.parse_args().command
    if command == 'audit':
        write(OUT / 'source-audit.json', source_audit())
    else:
        print(globals()[command]())
