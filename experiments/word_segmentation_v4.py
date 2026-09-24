"""Word segmentation v4 development: historical spelling model for unknown words.

    python -m experiments.word_segmentation_v4 develop
See experiments/word-segmentation-v4/PROTOCOL.md.
"""
import json
import subprocess
import time

from experiments.historical_sources import verify as historical
from experiments.verse_sources import verify as verse
from experiments.word_segmentation_v2 import ROOT, BASE, read, write, grade
from experiments.word_segmentation_v3 import OUT as V3, classify
from voynich.data import digest
from voynich.decipher import normalize
from voynich.unknown_words import Spelling, SpellingSegmenter
from voynich.unknown_words_v4 import MixtureSegmenter

OUT = ROOT / 'experiments/word-segmentation-v4'
CANDIDATES = ('hist', 'mix')
CODE = ['voynich/unknown_words.py', 'voynich/unknown_words_v4.py', 'voynich/segmentation.py', 'experiments/word_segmentation_v4.py',
        'experiments/word_segmentation_v3.py', 'experiments/historical_sources.py', 'experiments/verse_sources.py']


def historical_types():
    words = set()
    for r in historical():
        if r['split'] == 'train': words.update(normalize(' '.join(r['paragraphs'])).split())
    for r in verse():
        if r['split'] == 'train': words.update(normalize(' '.join(r['lines'])).split())
    return sorted(words)


def segmenters(model, frozen):
    order = frozen['selected']['order']; rate = frozen['unknown_rate']; elision = frozen['selected']['elision']; p = frozen['parameters']
    lexicon, hist = Spelling(model['lexicon'], order), Spelling(historical_types(), order)
    return dict(hist=SpellingSegmenter(model, hist, rate, elision, **p),
                mix=MixtureSegmenter(model, [lexicon, hist], [.5, .5], rate, elision, **p)), hist


def choose(rows):
    base = rows[0]['grades']
    def historical_mean(row): return (row['grades']['historical']['wer'] + row['grades']['verse']['wer']) / 2
    eligible = [r for r in rows[1:] if historical_mean(rows[0]) - historical_mean(r) >= .02
                and all(r['grades'][s]['wer'] - base[s]['wer'] <= .01 for s in ('historical', 'verse', 'modern'))]
    return min(eligible, key=lambda r: (historical_mean(r), r['grades']['modern']['wer'], r['candidate'] != 'mix'))['candidate'] if eligible else None


def develop():
    if (OUT / 'development.json').exists(): raise FileExistsError('Development recorded')
    model = read(BASE); lexicon = set(model['lexicon'])
    v3 = read(V3 / 'development.json'); frozen = read(V3 / 'freeze.json'); refs = v3['references']
    selected = next(r for r in v3['rows'] if r['candidate'] == frozen['selected'])
    rows = [dict(candidate='v3', grades={s: grade(selected['predictions'][s], r) for s, r in refs.items()}, predictions=selected['predictions'])]
    if rows[0]['grades'] != selected['grades']: raise ValueError('v3 grade drift')
    methods, hist = segmenters(model, frozen)
    for name in CANDIDATES:
        started = time.perf_counter()
        predictions = {s: methods[name].segment(r.replace(' ', '')) for s, r in refs.items()}
        rows.append(dict(candidate=name, predictions=predictions, seconds=time.perf_counter() - started,
                         grades={s: grade(predictions[s], r) for s, r in refs.items()}))
        print(json.dumps(dict(candidate=name, wer={s: round(g['wer'], 4) for s, g in rows[-1]['grades'].items()})), flush=True)
    for row in rows:
        row['diagnosis'] = {s: {k: v for k, v in classify(row['predictions'][s], r, lexicon).items() if k not in ('top_splits', 'top_merges')}
                            for s, r in refs.items()}
    write(OUT / 'development.json', dict(selected=choose(rows), rows=rows, references=refs, historical_types=len(historical_types()),
          historical_spelling_sha256=hist.digest(), v3_freeze_sha256=digest(V3 / 'freeze.json'), protocol_sha256=digest(OUT / 'PROTOCOL.md'),
          protocol_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'experiments/word-segmentation-v4/PROTOCOL.md'], cwd=ROOT, text=True).strip(),
          code_sha256={p: digest(ROOT / p) for p in CODE}, development_only=True, voynich_used=False, fresh_text_used=False))
    print('Selected:', read(OUT / 'development.json')['selected'], flush=True)


if __name__ == '__main__':
    develop()
