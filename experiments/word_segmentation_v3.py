"""Word-segmentation v3 groundwork: rubric-free extraction and a boundary-error diagnosis.

Development diagnosis only. Reads the saved v2 predictions; no segmenter setting is
chosen here and no fresh text is prepared. Released Villani/VIT records are described
separately and must never drive a later selection. CPU only.
"""
import argparse
from collections import Counter
import re

from bs4 import BeautifulSoup
from experiments.word_segmentation_v2 import ROOT, STATE, BASE, read, write, grade
from experiments.word_segmentation_fresh import extract_book
from voynich.segmentation import Segmenter, boundaries

V2 = ROOT / 'experiments/word-segmentation-v2'
OUT = ROOT / 'experiments/word-segmentation-v3'
ROMAN = re.compile(r'^[IVXLCDM]+$')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['check', 'diagnose'])
    if parser.parse_args().command == 'check':
        write(OUT / 'extraction-check.json', extraction_check())
        print(read(OUT / 'extraction-check.json'))
    else:
        diagnose()
