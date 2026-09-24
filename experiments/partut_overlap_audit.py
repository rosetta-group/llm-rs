"""Post-grading audit: ParTUT sentences that appear verbatim in UD_Italian-ISDT, per released modern passage.

The 20-word overlap check used when building passages misses sentences shorter than 20 words. ISDT train
fits the segmenter's word counts and the character prior, so verbatim ISDT-train sentences are training
leakage. Writes experiments/partut-overlap-audit.json.
"""
from collections import Counter
import json
from pathlib import Path

from experiments.segmentation import corpus
from voynich.corpora import conllu_sentences
from voynich.decipher import normalize

ROOT = Path(__file__).resolve().parents[1]
TESTS = {'word-segmentation-v3-fresh': [('test', 'artifacts/word-segmentation-v3-fresh/sources/it_partut-ud-test.conllu', 'partut-test:'),
                                        ('dev', 'artifacts/word-segmentation-v3-fresh/sources/it_partut-ud-dev.conllu', 'partut-dev:')],
         'joint-recovery-v6': [('train', 'artifacts/joint-recovery-v6/sources/it_partut-ud-train.conllu', 'partut-train:')],
         'word-segmentation-v4-fresh': [('train', 'artifacts/joint-recovery-v6/sources/it_partut-ud-train.conllu', 'partut-train:')]}


def isdt_split():
    out = {}
    for split in ('train', 'dev', 'test'):
        rows, _ = corpus('UD_Italian-ISDT', split)
        for r in rows: out.setdefault(normalize(' '.join(r['words'])), split)
    return out


def main():
    split = isdt_split(); result = {}
    for test, files in TESTS.items():
        text = {}
        for _, path, prefix in files:
            for r in conllu_sentences(ROOT / path): text[prefix + r['id']] = normalize(' '.join(r['words']))
        cases = []
        for a in json.loads((ROOT / f'experiments/{test}/evaluated-records.json').read_text())['answers']:
            if a['dataset'] != 'modern': continue
            letters = Counter()
            for i in a['source_ids']:
                t = text[i.split(':part')[0]]; letters[split.get(t, 'not in ISDT')] += len(t.replace(' ', ''))
            total = sum(letters.values())
            cases.append(dict(id=a['id'], sentences=len(a['source_ids']), letter_share={k: v / total for k, v in letters.items()}))
        pooled = Counter()
        for c, a in zip(cases, [x for x in json.loads((ROOT / f'experiments/{test}/evaluated-records.json').read_text())['answers'] if x['dataset'] == 'modern']):
            pooled.update({k: v * len(a['plaintext'].replace(' ', '')) for k, v in c['letter_share'].items()})
        total = sum(pooled.values())
        result[test] = dict(cases=cases, pooled_letter_share={k: v / total for k, v in pooled.items()},
                            isdt_train_share=pooled.get('train', 0) / total)
    train_rows = [normalize(' '.join(r['words'])) for r in conllu_sentences(ROOT / TESTS['joint-recovery-v6'][0][1])]
    result['partut_train_sentences'] = len(train_rows)
    result['partut_train_in_isdt'] = sum(t in split for t in train_rows)
    result['partut_train_in_isdt_train'] = sum(split.get(t) == 'train' for t in train_rows)
    (ROOT / 'experiments/partut-overlap-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: (v['pooled_letter_share'] if isinstance(v, dict) else v) for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
