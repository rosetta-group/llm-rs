"""Development only: the confirmed rule and the candidate rule at Voynich-like pairing (RESPACING 9).

python -m experiments.respacing9_development prepare | run | evaluate

The 48 released passages of the second confirmation are encrypted again at RESPACING 9 (about 75%
of letters in pairs, the only Naibbe regime that matches the manuscript's near-duplicate rate), with
new keys and encoder seeds, same block pairing and the same three inputs per block. Decoder A fits
all eight v2 priors. Two rules are scored on the same fits:

  current    decide_transfer, per-run transfer score, eight separate languages, ceiling 0.50
  candidate  decide_transfer, one length code per passage, Catalan and Occitan grouped, ceiling 0.50
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

from experiments import key_recovery_confirmation_v2 as v2
from experiments import key_recovery_development as development
from experiments.joint_development import load_vendor
from experiments.rejection_transfer_v2 import encrypt, read, seal
from voynich.decipher import ALPHABET, edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer, frequency_copy
from voynich.rejection_v3 import group_scores, label, rescore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/respacing9-development'
STATE = ROOT / 'artifacts/respacing9-development'
RESPACING = 9
KINDS = ('positive', 'shuffle', 'frequency_copy')


def prepare():
    if (STATE / 'challenge.json').exists():
        raise FileExistsError('Already prepared')
    released = read(v2.STATE / 'evaluator-only/answers.json')
    blocks = sorted({a['block']: a for a in released.values() if a['kind'] == 'positive'}.items())
    vendor = load_vendor()
    vendor.RESPACING = RESPACING
    rng = random.SystemRandom()
    answers, schedule = {}, []
    for block, source in blocks:
        pair = source['passages']
        key_seed = rng.randrange(2 ** 63)
        letters = list(ALPHABET)
        random.Random(key_seed).shuffle(letters)
        key = dict(zip(ALPHABET, letters))
        seeds = [rng.randrange(2 ** 63) for _ in range(6)]
        positive = [encrypt(vendor, p['plaintext'], key, s) for p, s in zip(pair, seeds[:2])]
        shuffled = [list(t) for t in positive]
        for tokens, seed in zip(shuffled, seeds[2:4]):
            random.Random(seed).shuffle(tokens)
        frequency = [frequency_copy(t, seed) for t, seed in zip(positive, seeds[4:6])]
        ids = []
        for kind, pair_tokens in zip(KINDS, (positive, shuffled, frequency)):
            ident = hashlib.sha256(str(rng.randrange(2 ** 128)).encode()).hexdigest()[:20]
            ids.append(ident)
            for role, tokens in zip(('fit', 'transfer'), pair_tokens):
                seal(STATE / 'public' / f'{ident}-{role}.json', dict(id=ident, ciphertext=' '.join(tokens)))
            answers[ident] = dict(language=source['language'], kind=kind, block=block, passages=pair,
                                  key_seed=key_seed, seeds=seeds, key=key, respacing=RESPACING)
        schedule.append(ids)
    seal(STATE / 'evaluator-only/answers.json', answers)
    seal(STATE / 'challenge.json', dict(schedule=schedule, respacing=RESPACING, at=time.time()))
    print(f'Prepared {len(schedule)} blocks at RESPACING {RESPACING}.', flush=True)


def worker(job):
    ident, model = job
    path = STATE / 'fit' / f'{ident}-{model}.json'
    if path.exists():
        return str(path)
    prior = CharacterPrior.load(v2.prior_path(model))
    fitted = development.fit_both(read(STATE / 'public' / f'{ident}-fit.json')['ciphertext'].split(),
                                  prior, read(v2.PRIOR_FREEZE))
    held = read(STATE / 'public' / f'{ident}-transfer.json')['ciphertext'].split()
    for arm in ('B', 'A'):
        fitted[arm]['transfer'] = transfer(held, fitted[arm]['mapping'], prior)
    seal(path, dict(id=ident, model=model, result=fitted))
    return str(path)


def run(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    models = list(read(v2.OUT / 'freeze.json')['labels'].values())
    jobs = [(i, m) for block in read(STATE / 'challenge.json')['schedule'] for i in block for m in models]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, _ in enumerate(pool.map(worker, jobs), 1):
            if done % 24 == 0:
                print(f'{done}/{len(jobs)} fits', flush=True)
    print('Fitted', len(jobs), 'input-model pairs.', flush=True)


def scores_for(fits, labels, entropy, priors, arm, honest):
    out = {}
    for lang, model in labels.items():
        r = fits[model][arm]
        held = rescore(r['transfer'], priors[model]) if honest else r['transfer']
        out[lang] = dict(fit_excess=r['bits_per_letter'] - entropy[model],
                         transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - entropy[model],
                         coverage=held['token_coverage'], cap_hit=r['cap_hit'])
    return out


def evaluate():
    freeze = read(v2.OUT / 'freeze.json')
    labels, entropy = freeze['labels'], freeze['entropy']
    priors = {m: CharacterPrior.load(v2.prior_path(m)) for m in labels.values()}
    answers = read(STATE / 'evaluator-only/answers.json')
    rules = dict(current=dict(honest=False, grouped=False), candidate=dict(honest=True, grouped=True),
                 honest_only=dict(honest=True, grouped=False), grouped_only=dict(honest=False, grouped=True))
    rows = []
    for block in read(STATE / 'challenge.json')['schedule']:
        for ident in block:
            answer = answers[ident]
            fits = {m: read(STATE / 'fit' / f'{ident}-{m}.json')['result'] for m in labels.values()}
            row = dict(id=ident, block=answer['block'], language=answer['language'], kind=answer['kind'])
            for arm in ('A', 'B'):
                for name, rule in rules.items():
                    s = scores_for(fits, labels, entropy, priors, arm, rule['honest'])
                    truth = label(answer['language']) if rule['grouped'] else answer['language']
                    if rule['grouped']:
                        s = group_scores(s)
                    decision = dict(full=decide_transfer(s))
                    if answer['kind'] == 'positive':
                        decision['omitted'] = decide_transfer(s, [l for l in s if l != truth])
                    decision['truth'] = truth
                    row[f'{arm}_{name}'] = decision
                if answer['kind'] == 'positive':
                    model = labels[answer['language']]
                    texts = [p['plaintext'] for p in answer['passages']]
                    row[f'{arm}_transfer_cer'] = edit_distance(fits[model][arm]['transfer']['recovered'], texts[1]) / len(texts[1])
            rows.append(row)
    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] != 'positive']
    summary = {}
    for key in [k for k in rows[0] if k.split('_', 1)[1] in rules]:
        summary[key] = dict(correct=sum(r[key]['full']['accepted'] == r[key]['truth'] for r in positives),
                            wrong=sum(r[key]['full']['accepted'] not in (None, r[key]['truth']) for r in positives),
                            omitted=sum(r[key]['omitted']['accepted'] is not None for r in positives),
                            negatives=sum(r[key]['full']['accepted'] is not None for r in negatives),
                            inconclusive=sum(r[key]['full']['inconclusive'] for r in rows))
    summary['median_transfer_cer_A'] = sorted(r['A_transfer_cer'] for r in positives)[len(positives) // 2]
    seal(OUT / 'results.json', dict(rows=rows, summary=summary, respacing=RESPACING, development_only=True,
                                    released_passages=True, at=time.time()))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    run(args.workers) if args.command == 'run' else globals()[args.command]()
