"""Development only: broadened Latin, German and Catalan priors on the released round-one confirmation.

python -m experiments.key_recovery_v2_development priors | run | evaluate

The 72 round-one inputs are released. Fits under the five unchanged priors are reused from the
round-one records; only the three new priors are fitted, with the same decoder (A and B arms).
Keys are fixed before transfer. The decision rule is the frozen `decide_transfer`.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from experiments import key_recovery_development as development
from experiments.rejection_transfer_v2 import read, seal
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/key-recovery-v2-development'
STATE = ROOT / 'artifacts/key-recovery-v2-development'
PARTITIONS = ROOT / 'artifacts/key-recovery-confirmation-v2/partitions.json'
PRIOR_FREEZE = ROOT / 'experiments/language-expansion/freeze.json'
ROUND_ONE = ROOT / 'artifacts/key-recovery-confirmation'
NEW = dict(latin='latin_broad2', german='german_broad2', catalan='catalan_broad2')


def labels():
    """Candidate label -> model: three new priors, five unchanged."""
    return {l: NEW.get(l, m) for l, m in read(PRIOR_FREEZE)['expanded'].items()}


def priors():
    parts = read(PARTITIONS)['priors']
    out = {}
    for model in NEW.values():
        path = STATE / 'priors' / f'{model}.npz'
        if path.exists():
            raise FileExistsError(path)
        train = [r['text'] for r in parts[model]['train']]
        if sum(len(t.replace(' ', '')) for t in train) != 400_000:
            raise ValueError('Unmatched prior budget: ' + model)
        prior = CharacterPrior.fit(train)
        path.parent.mkdir(parents=True, exist_ok=True)
        prior.save(path)
        held = ''.join(r['text'] for r in parts[model]['calibration']).replace(' ', '')
        out[model] = dict(entropy=prior.bits(held) / len(held), sha256=digest(path))
    seal(OUT / 'priors.json', dict(models=out, partitions_sha256=digest(PARTITIONS), at=time.time()))
    print(json.dumps(out, indent=1))


def inputs():
    return [i for block in read(ROUND_ONE / 'challenge.json')['schedule'] for i in block]


def worker(job):
    ident, model = job
    path = STATE / 'fit' / f'{ident}-{model}.json'
    if path.exists():
        return str(path)
    record = read(OUT / 'priors.json')['models'][model]
    prior_path = STATE / 'priors' / f'{model}.npz'
    if digest(prior_path) != record['sha256']:
        raise ValueError('Prior changed')
    prior = CharacterPrior.load(prior_path)
    public = ROUND_ONE / 'public'
    fitted = development.fit_both(read(public / f'{ident}-fit.json')['ciphertext'].split(), prior, read(PRIOR_FREEZE))
    held = read(public / f'{ident}-transfer.json')['ciphertext'].split()
    for arm in ('B', 'A'):
        fitted[arm]['transfer'] = transfer(held, fitted[arm]['mapping'], prior)
    seal(path, dict(id=ident, model=model, result=fitted, priors_sha256=digest(OUT / 'priors.json')))
    return str(path)


def run(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    jobs = [(i, m) for i in inputs() for m in NEW.values()]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, _ in enumerate(pool.map(worker, jobs), 1):
            if done % 24 == 0:
                print(f'{done}/{len(jobs)} fits', flush=True)
    print('Fitted', len(jobs), 'input-model pairs.', flush=True)


def arm_result(ident, model, arm):
    """(fit result, transfer result) for one arm; unchanged priors come from round one."""
    if model in NEW.values():
        r = read(STATE / 'fit' / f'{ident}-{model}.json')['result'][arm]
        return r, r['transfer']
    return (read(ROUND_ONE / 'fit' / f'{ident}-{model}.json')['result'][arm],
            read(ROUND_ONE / 'transfer' / f'{ident}-{model}.json')['result'][arm])


def evaluate():
    entropy = dict(read(PRIOR_FREEZE)['entropy'], **{m: r['entropy'] for m, r in read(OUT / 'priors.json')['models'].items()})
    candidates = labels()
    answers = read(ROUND_ONE / 'evaluator-only/answers.json')
    rows = []
    for ident in inputs():
        answer = answers[ident]
        row = dict(id=ident, block=answer['block'], language=answer['language'], kind=answer['kind'])
        for arm in ('B', 'A'):
            scores = {}
            for label, model in candidates.items():
                fit, held = arm_result(ident, model, arm)
                scores[label] = dict(fit_excess=fit['bits_per_letter'] - entropy[model],
                                     transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - entropy[model],
                                     coverage=held['token_coverage'], cap_hit=fit['cap_hit'])
            row[arm] = dict(scores=scores, full=decide_transfer(scores))
            if answer['kind'] == 'positive':
                row[arm]['omitted'] = decide_transfer(scores, [l for l in candidates if l != answer['language']])
                fit, held = arm_result(ident, candidates[answer['language']], arm)
                texts = [p['plaintext'] for p in answer['passages']]
                row[arm]['transfer_cer'] = edit_distance(held['recovered'], texts[1]) / len(texts[1])
        rows.append(row)
    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] != 'positive']
    summary = {arm: dict(correct=sum(r[arm]['full']['accepted'] == r['language'] for r in positives),
                         wrong_language=sum(r[arm]['full']['accepted'] not in (None, r['language']) for r in positives),
                         omitted_accepted=sum(r[arm]['omitted']['accepted'] is not None for r in positives),
                         negatives_accepted=sum(r[arm]['full']['accepted'] is not None for r in negatives),
                         inconclusive=sum(r[arm]['full']['inconclusive'] for r in rows),
                         by_language={l: sum(r[arm]['full']['accepted'] == l for r in positives if r['language'] == l)
                                      for l in candidates})
               for arm in ('B', 'A')}
    seal(OUT / 'results.json', dict(rows=rows, summary=summary, candidates=candidates, development_only=True,
                                    voynich_used=False, at=time.time()))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['priors', 'run', 'evaluate'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.workers)
    else:
        globals()[args.command]()
