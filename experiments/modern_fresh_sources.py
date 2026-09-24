"""Pin a fresh modern Italian source (UD_Italian-PUD test) and audit it against every Italian text already used.

A sentence is eligible for future fresh tests only if it (1) does not appear verbatim, after `normalize`, in any
split of UD_Italian-ISDT, ParTUT or VIT, or in any released passage, and (2) shares no 20-word sequence with
fitting, development or released text. The verbatim rule applies at any length (the 20-word rule alone
missed ParTUT train's short ISDT sentences).

    python -m experiments.modern_fresh_sources
"""
import json
from pathlib import Path

from experiments.segmentation import corpus
from experiments.word_segmentation_fresh import fetch, seen_ngrams
from experiments.word_segmentation_v3_fresh import released_ngrams
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import normalize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'artifacts/modern-fresh/sources'
MANIFEST = ROOT / 'experiments/modern-fresh-sources.json'
REPO = 'UniversalDependencies/UD_Italian-PUD'
OTHER = {'partut-test': 'artifacts/word-segmentation-v3-fresh/sources/it_partut-ud-test.conllu',
         'partut-dev': 'artifacts/word-segmentation-v3-fresh/sources/it_partut-ud-dev.conllu',
         'partut-train': 'artifacts/joint-recovery-v6/sources/it_partut-ud-train.conllu',
         'vit-test': 'artifacts/word-segmentation-v2/sources/it_vit-ud-test.conllu'}
RELEASED = ['experiments/word-segmentation-v2/evaluated-records.json', 'experiments/word-segmentation-v3-fresh/evaluated-records.json',
            'experiments/joint-recovery-v5/evaluated-records.json', 'experiments/joint-recovery-v6/evaluated-records.json',
            'experiments/word-segmentation-v4-fresh/evaluated-records.json', 'experiments/language-id/evaluated-records.json']


def known_sentences():
    known = {}
    for split in ('train', 'dev', 'test'):
        rows, _ = corpus('UD_Italian-ISDT', split)
        for r in rows: known.setdefault(normalize(' '.join(r['words'])), 'isdt-' + split)
    for name, path in OTHER.items():
        for r in conllu_sentences(ROOT / path): known.setdefault(normalize(' '.join(r['words'])), name)
    return known


def released_text():
    grams, plain = set(), []
    for path in RELEASED:
        for r in json.loads((ROOT / path).read_text())['answers']:
            w = r['plaintext'].split(); plain.append(r['plaintext']); grams.update(tuple(w[i:i + 20]) for i in range(len(w) - 19))
    return grams, plain


def main():
    if MANIFEST.exists(): raise FileExistsError('Already pinned')
    files = [fetch(f'https://api.github.com/repos/{REPO}/commits/master', RAW / 'pud-commit.json')]
    revision = json.loads((RAW / 'pud-commit.json').read_text())['sha']
    for name in ('it_pud-ud-test.conllu', 'README.md', 'LICENSE.txt'):
        r = fetch(f'https://raw.githubusercontent.com/{REPO}/{revision}/{name}', RAW / name)
        r.update(revision=revision, repository='UD_Italian-PUD', license='CC BY-SA 3.0'); files.append(r)
    known = known_sentences(); grams, plain = released_text(); seen = seen_ngrams() | released_ngrams() | grams
    joined = ' '.join(plain)
    rows, eligible, reasons = [], [], {}
    for r in conllu_sentences(RAW / 'it_pud-ud-test.conllu'):
        text = normalize(' '.join(r['words'])); w = text.split()
        why = known.get(text) or ('released-passage' if f' {text} ' in f' {joined} ' and len(w) >= 5 else None)
        if not why and any(tuple(w[i:i + 20]) in seen for i in range(len(w) - 19)): why = 'shared-20-gram'
        if why: reasons[why] = reasons.get(why, 0) + 1
        else: eligible.append(r['id'])
        rows.append(dict(id=r['id'], letters=len(text.replace(' ', ''))))
    letters = {x['id']: x['letters'] for x in rows}
    MANIFEST.write_text(json.dumps(dict(
        files=files, sentences=len(rows), eligible_sentences=len(eligible), eligible_letters=sum(letters[i] for i in eligible),
        excluded=reasons, eligible_ids=eligible,
        rule='eligible = not verbatim (normalized) in ISDT train/dev/test, ParTUT test/dev/train, VIT test or any released passage, and no shared 20-word sequence with fitting, development or released text',
        genre='news and Wikipedia sentences, translated into Italian (mostly from English) for the CoNLL 2017 parallel treebank',
        attribution='UD_Italian-PUD: Google, UD contributors; CC BY-SA 3.0'), indent=2) + '\n')
    print(dict(sentences=len(rows), eligible=len(eligible), eligible_letters=sum(letters[i] for i in eligible), excluded=reasons))


if __name__ == '__main__':
    main()
