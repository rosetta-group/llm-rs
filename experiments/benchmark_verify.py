"""Read-only audit of frozen methods, predictions, splits, and public case fields."""
import json
from pathlib import Path

from experiments.segmentation import check_freeze, corpus
from experiments.codebook_free import verify
from voynich.data import digest
from voynich.decipher import normalize

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    segmenter=check_freeze();decoder=verify();association=read('experiments/association/freeze.json')
    assert segmenter['protocol_sha256']==digest(ROOT/'experiments/METHOD_BENCHMARK_PLAN.md')
    for key,path in [('code_sha256','experiments/association.py'),('protocol_sha256','experiments/association/PROTOCOL.md'),
                     ('metadata_sha256','artifacts/data/gc/documents.json'),('split_sha256','artifacts/data/gc/manifest.json')]:
        assert association[key]==digest(ROOT/path),path
    assert association['source']['sha256']==digest(ROOT/association['source']['path'])
    old=read('artifacts/decipherment/evaluator-only/answers.json')
    segments=read('artifacts/segmentation/evaluator-only/answers.json')
    challenges=read('artifacts/codebook-free/evaluator-only/answers.json')
    old_ids={s for r in old for s in r['source_sentences']}
    segment_ids={k:{s for r in segments if r['dataset']==k for s in r['source_ids']} for k in ('modern','historical')}
    assert not old_ids & segment_ids['historical']
    passages={r['passage']:r for r in challenges}
    prior=set()
    for split in ('train','dev'):
        rows,_=corpus('UD_Italian-ISDT',split)
        prior.update(normalize(' '.join(r['words'])) for r in rows)
    for dataset,repository,split in [('modern','UD_Italian-ISDT','test'),('historical','UD_Italian-Old','train')]:
        ids=[s for r in passages.values() if r['dataset']==dataset for s in r['source_ids']]
        assert len(ids)==len(set(ids))
        assert not set(ids) & segment_ids[dataset]
        if dataset=='historical':assert not set(ids) & old_ids
        selected=set(ids)|segment_ids[dataset]
        rows,_=corpus(repository,split)
        assert not any(normalize(' '.join(r['words'])) in prior for r in rows if r['id'] in selected)
    assert all(set(r)=={'id','ciphertext'} for r in read('artifacts/codebook-free/public.json'))
    for name in ('segmentation','codebook-free'):
        results=read(f'experiments/{name}/results.json');challenge=read(f'artifacts/{name}/challenge.json')
        assert results['predictions_sha256']==digest(ROOT/f'artifacts/{name}/predictions.json')
        assert challenge['public_sha256']==digest(ROOT/f'artifacts/{name}/public.json')
        assert challenge['answers_sha256']==digest(ROOT/f'artifacts/{name}/evaluator-only/answers.json')
    challenge=read('artifacts/codebook-free/challenge.json');predictions=read('artifacts/codebook-free/predictions.json')
    assert decoder['at']<=challenge['at']<=predictions['at']
    results=read('experiments/codebook-free/results.json')
    assert all(c['proposals']==72000 and not c['cap_hit'] for r in results['cases'] for c in r['candidates'])
    print('Verified frozen hashes, predictions, fresh-passage exclusions, public-only case schema, and completed search budgets')


if __name__=='__main__':main()
