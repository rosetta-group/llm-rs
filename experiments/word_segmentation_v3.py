"""Word-segmentation v3: rubric-free extraction, boundary diagnosis, unknown-word model.

`check` and `diagnose` read saved v2 records only. `develop` scores the four declared
candidates of PROTOCOL.md on development streams; `freeze` records the selection before
any fresh source is named. Released Villani/VIT records never drive a selection. CPU only.
"""
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
import time

from bs4 import BeautifulSoup
from experiments.word_segmentation_v2 import ROOT, STATE, BASE, read, write, grade
from experiments.word_segmentation_fresh import extract_book
from voynich.data import digest
from voynich.decipher import normalize
from voynich.segmentation import Segmenter, boundaries
from voynich.unknown_words import Spelling, SpellingSegmenter, unknown_rate

V2 = ROOT / 'experiments/word-segmentation-v2'
OUT = ROOT / 'experiments/word-segmentation-v3'
ROMAN = re.compile(r'^[IVXLCDM]+$')
CANDIDATES = [dict(order=3, elision=False), dict(order=3, elision=True),
              dict(order=5, elision=False), dict(order=5, elision=True)]
CODE = ['voynich/unknown_words.py', 'voynich/segmentation.py', 'voynich/decipher.py', 'voynich/corpora.py',
        'voynich/data.py', 'experiments/word_segmentation_v3.py', 'experiments/word_segmentation_v2.py',
        'experiments/word_segmentation_fresh.py', 'experiments/segmentation_audit.py',
        'experiments/segmentation.py', 'experiments/historical_sources.py']


def is_rubric(text):
    """Wikisource chapter rubric stored as <p>: first non-empty line is a bare Roman numeral."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return len(lines) >= 2 and bool(ROMAN.match(lines[0]))


def extract_book_v3(html):
    """The v2 extractor applied after removing paragraph-encoded chapter rubrics."""
    soup = BeautifulSoup(html, 'html.parser')
    root = soup.select_one('#box_esterno')
    if root is None: raise ValueError('Missing Wikisource text container')
    removed = [p.extract().get_text() for p in root.select('p') if is_rubric(p.get_text())]
    return extract_book(str(soup)), removed


def extraction_check():
    """Check v3 on the released Villani source: only the audited rubrics disappear."""
    html = (STATE / 'sources/villani-book1.html').read_text()
    old = extract_book(html)
    new, removed = extract_book_v3(html)
    audit = read(V2 / 'source-audit.json')
    old_texts = [r['text'] for r in old]
    new_texts = [r['text'] for r in new]
    kept = [t for t in old_texts if t in set(new_texts)]
    if kept != new_texts: raise ValueError('v3 changed or reordered a body paragraph')
    dropped = len(old_texts) - len(new_texts)
    if dropped != audit['heading_paragraphs']:
        raise ValueError('v3 removed %d paragraphs; audit found %d rubrics' % (dropped, audit['heading_paragraphs']))
    return dict(source='artifacts/word-segmentation-v2/sources/villani-book1.html', released_source=True,
                v2_paragraphs=len(old), v3_paragraphs=len(new), rubric_paragraphs_removed=len(removed),
                rubrics_under_extractor_minimum=len(removed)-dropped, dropped_extracted_paragraphs=dropped,
                audit_heading_paragraphs=audit['heading_paragraphs'], body_paragraphs_unchanged=True,
                rubric_words_max=max(len(t.split()) for t in removed),
                body_words_min=min(len(t.split()) for t in new_texts))


def spans(text):
    out, offset = [], 0
    for word in text.split():
        out.append((offset, offset + len(word), word)); offset += len(word)
    return out


def classify(prediction, reference, lexicon):
    """Attribute every wrong boundary to a reference word (extra) or a predicted token (missing)."""
    grade(prediction, reference)
    n = len(reference.replace(' ', ''))
    gold, pred = boundaries(reference), boundaries(prediction)
    words = Counter(); extra = Counter(); missing = Counter()
    split_examples = Counter(); merge_examples = Counter()
    for start, end, word in spans(reference):
        inside = [b for b in pred if start < b < end]
        edges = (start == 0 or start in pred) and (end == n or end in pred)
        known = word in lexicon
        kind = ('correct' if not inside else 'split') if edges else ('merged' if not inside else 'misaligned')
        words[kind] += 1
        if inside:
            extra['known_word' if known else 'missing_form'] += len(inside)
            cuts = [start] + sorted(inside) + [end]
            split_examples[(word, known, ' '.join(word[a-start:b-start] for a, b in zip(cuts, cuts[1:])))] += 1
    for start, end, token in spans(prediction):
        inside = [b for b in gold if start < b < end]
        if inside:
            missing['known_token' if token in lexicon else 'unknown_token'] += len(inside)
            cuts = [start] + sorted(inside) + [end]
            merge_examples[(token, token in lexicon, ' '.join(token[a-start:b-start] for a, b in zip(cuts, cuts[1:])))] += 1
    oov = [w for _, _, w in spans(reference) if w not in lexicon]
    return dict(words=dict(words), reference_words=sum(words.values()),
                extra_spaces=dict(extra), extra_total=sum(extra.values()),
                missing_spaces=dict(missing), missing_total=sum(missing.values()),
                oov_tokens=len(oov), oov_types=len(set(oov)),
                oov_tokens_split=sum(n for (w, k, _), n in split_examples.items() if not k),
                top_splits=[dict(reference=w, known=k, predicted=p, count=c) for (w, k, p), c in split_examples.most_common(15)],
                top_merges=[dict(predicted=t, known=k, reference=r, count=c) for (t, k, r), c in merge_examples.most_common(10)])


def oracle(model, parameters, reference, forms, counts):
    """Labelled oracle: the frozen segmenter with the stream's own missing forms added."""
    lexicon = set(model['lexicon'])
    augmented = dict(model, lexicon=sorted(lexicon | forms), counts=dict(model['counts']))
    if counts:
        for word, c in Counter(w for w in reference.split() if w in forms).items():
            augmented['counts'][word] = augmented['counts'].get(word, 0) + c
    return grade(Segmenter(augmented, **parameters).segment(reference.replace(' ', '')), reference)


def diagnose():
    model = read(BASE); lexicon = set(model['lexicon'])
    dev = read(V2 / 'development.json'); parameters = dev['parameters']
    baseline = read(V2 / 'dev-weight-0.json')
    if baseline['weight'] != 0: raise ValueError('Expected the frozen baseline predictions')
    development = {}
    for stream, reference in dev['references'].items():
        prediction = baseline['predictions'][stream]
        if grade(prediction, reference) != baseline['grades'][stream]: raise ValueError('Saved grade drift: ' + stream)
        forms = {w for w in reference.split() if w not in lexicon}
        development[stream] = dict(baseline=baseline['grades'][stream], diagnosis=classify(prediction, reference, lexicon),
                                   oracle_forms_only=oracle(model, parameters, reference, forms, False),
                                   oracle_forms_and_counts=oracle(model, parameters, reference, forms, True))
    release = read(V2 / 'evaluated-records.json')
    predictions = {r['id']: r['baseline'] for r in release['predictions']['rows']}
    released = {}
    for dataset in ('historical', 'modern'):
        rows = [r for r in release['answers'] if r['dataset'] == dataset]
        parts = [classify(predictions[r['id']], r['plaintext'], lexicon) for r in rows]
        released[dataset] = dict(cases=len(rows), **{k: sum(p[k] for p in parts) for k in
            ('reference_words', 'extra_total', 'missing_total', 'oov_tokens', 'oov_tokens_split')},
            extra_spaces=dict(sum((Counter(p['extra_spaces']) for p in parts), Counter())),
            missing_spaces=dict(sum((Counter(p['missing_spaces']) for p in parts), Counter())))
    result = dict(development=development, released_descriptive=released, model_sha256=dev['base_sha256'],
                  parameters=parameters, oracle_note='Oracles insert each stream\'s own missing forms; ceilings, never a method.',
                  released_note='Villani/VIT are released; descriptive only, never for selection.',
                  development_only=True, voynich_used=False, fresh_text_used=False)
    write(OUT / 'diagnosis.json', result)
    return result


def morphit_forms():
    lex = read(ROOT / 'experiments/segmentation-sources.json')['lexicon']
    if digest(ROOT / lex['path']) != lex['sha256']: raise ValueError('Lexicon drift')
    forms = {normalize(line.split('\t')[0]) for line in (ROOT / lex['path']).read_text(encoding='latin-1').splitlines()}
    return {w for w in forms if w and ' ' not in w}


def training_tokens():
    """Unweighted ISDT train and historical prose train, as used to fit the base model."""
    from experiments.historical_sources import verify as historical
    from experiments.segmentation import corpus
    train, _ = corpus('UD_Italian-ISDT', 'train')
    texts = [normalize(' '.join(r['words'])) for r in train]
    texts += [normalize(' '.join(r['paragraphs'])) for r in historical() if r['split'] == 'train']
    return [w for t in texts for w in t.split()]


def components(model):
    """Everything fitted from training data only: the unknown rate and one spelling model per order."""
    rate = unknown_rate(training_tokens(), morphit_forms())
    spellings = {order: Spelling(model['lexicon'], order) for order in sorted({c['order'] for c in CANDIDATES})}
    return rate, spellings


def choose(rows):
    """v2 selection rule: >=3-point historical mean gain, no stream worse by >1 point."""
    base = rows[0]['grades']
    def historical(row): return sum(row['grades'][s]['wer'] for s in ('historical', 'verse')) / 2
    eligible = [r for r in rows[1:] if historical(rows[0]) - historical(r) >= .03
                and all(r['grades'][s]['wer'] - base[s]['wer'] <= .01 for s in ('historical', 'verse', 'modern'))]
    if not eligible: return None
    best = min(eligible, key=lambda r: (historical(r), r['grades']['modern']['wer'], r['candidate']['order'], r['candidate']['elision']))
    return best['candidate']


def develop():
    if (OUT / 'development.json').exists(): raise FileExistsError('Development recorded')
    model = read(BASE); lexicon = set(model['lexicon'])
    v2 = read(V2 / 'development.json'); parameters = v2['parameters']
    baseline = read(V2 / 'dev-weight-0.json'); refs = v2['references']
    rate, spellings = components(model)
    rows = [dict(candidate=None, predictions=baseline['predictions'],
                 grades={s: grade(baseline['predictions'][s], r) for s, r in refs.items()})]
    for candidate in CANDIDATES:
        segmenter = SpellingSegmenter(model, spellings[candidate['order']], rate, candidate['elision'], **parameters)
        began = time.perf_counter()
        predictions = {s: segmenter.segment(r.replace(' ', '')) for s, r in refs.items()}
        rows.append(dict(candidate=candidate, predictions=predictions, seconds=time.perf_counter() - began,
                         grades={s: grade(predictions[s], r) for s, r in refs.items()}))
        print(json.dumps(dict(candidate=candidate, wer={s: round(g['wer'], 4) for s, g in rows[-1]['grades'].items()})), flush=True)
    for row in rows:
        row['diagnosis'] = {s: {k: v for k, v in classify(row['predictions'][s], r, lexicon).items()
                                if k not in ('top_splits', 'top_merges')} for s, r in refs.items()}
    write(OUT / 'development.json', dict(selected=choose(rows), rows=rows, references=refs, parameters=parameters,
          unknown_rate=rate, spelling_sha256={str(o): m.digest() for o, m in spellings.items()},
          base_sha256=digest(BASE), protocol_sha256=digest(OUT / 'PROTOCOL.md'),
          protocol_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'experiments/word-segmentation-v3/PROTOCOL.md'], cwd=ROOT, text=True).strip(),
          code_sha256={p: digest(ROOT / p) for p in CODE}, development_only=True, voynich_used=False, fresh_text_used=False))
    print('Selected:', read(OUT / 'development.json')['selected'], flush=True)


def freeze():
    if (OUT / 'freeze.json').exists(): raise FileExistsError('Already frozen')
    dev = read(OUT / 'development.json')
    if dev['selected'] is None: raise ValueError('No selected candidate')
    for p, h in dev['code_sha256'].items():
        if digest(ROOT / p) != h: raise ValueError('Development code drift: ' + p)
    if digest(OUT / 'PROTOCOL.md') != dev['protocol_sha256']: raise ValueError('Protocol drift')
    paths = CODE + ['experiments/word-segmentation-v3/PROTOCOL.md', 'experiments/word-segmentation-v3/development.json',
                    'experiments/segmentation-sources.json', 'experiments/standard-decipherment/development.json']
    write(OUT / 'freeze.json', dict(files={p: digest(ROOT / p) for p in paths}, baseline_sha256=digest(BASE),
          parameters=dev['parameters'], selected=dev['selected'], unknown_rate=dev['unknown_rate'],
          spelling_sha256=dev['spelling_sha256'][str(dev['selected']['order'])], version=1))
    print('Freeze written; commit before naming or fetching fresh sources')


def verify():
    """Check the committed freeze and refit the training-only components bit for bit."""
    f = read(OUT / 'freeze.json')
    for p, h in list(f['files'].items()) + [('experiments/word-segmentation-v3/freeze.json', digest(OUT / 'freeze.json'))]:
        if digest(ROOT / p) != h: raise ValueError('Frozen file drift: ' + p)
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)).hexdigest() != h:
            raise ValueError('Freeze input not committed: ' + p)
    if digest(BASE) != f['baseline_sha256']: raise ValueError('Baseline drift')
    model = read(BASE)
    if unknown_rate(training_tokens(), morphit_forms()) != f['unknown_rate']: raise ValueError('Unknown rate drift')
    if Spelling(model['lexicon'], f['selected']['order']).digest() != f['spelling_sha256']: raise ValueError('Spelling drift')
    dev = read(OUT / 'development.json')
    if choose(dev['rows']) != f['selected']: raise ValueError('Selection drift')
    for row in dev['rows']:
        for s, r in dev['references'].items():
            if grade(row['predictions'][s], r) != row['grades'][s]: raise ValueError('Development grade drift')
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'diagnose', 'develop', 'freeze', 'verify'])
    command = parser.parse_args().command
    if command == 'check':
        write(OUT / 'extraction-check.json', extraction_check())
        print(read(OUT / 'extraction-check.json'))
    elif command == 'verify':
        print(verify())
    else:
        globals()[command]()
