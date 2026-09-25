"""Frozen, paired baseline/expanded/omitted historical-language pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import time

from experiments.language_coverage_sources import ROOT, OUT, STATE
from experiments import rejection_followups_v2 as previous
from experiments.rejection_transfer_v2 import read, seal, encrypt
from experiments.joint_development import load_vendor
from voynich.data import digest
from voynich.decipher import ALPHABET, edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer

BASELINE = dict(latin='latin', german='german', old_french='old_french', english='english', italian='italian')
EXPANDED = dict(BASELINE, latin='latin_broad', german='german_broad', catalan='catalan')
MODELS = tuple(dict.fromkeys([*BASELINE.values(), *EXPANDED.values()]))


def decisions(scores, language):
    base = {l: scores[m] for l, m in BASELINE.items()}
    expanded = {l: scores[m] for l, m in EXPANDED.items()}
    return dict(baseline=decide_transfer(base), expanded=decide_transfer(expanded),
                omitted=decide_transfer(expanded, [l for l in expanded if l != language]))


def freeze():
    old = previous.verify()
    audit = read(OUT / 'sources.json')
    if digest(STATE / 'partitions.json') != audit['partitions_sha256']:
        raise ValueError('Partition drift')
    parts = read(STATE / 'partitions.json')
    prior_files, entropy, model_hashes = {}, {}, {}
    (STATE / 'priors').mkdir(exist_ok=True)
    for name in MODELS:
        texts = [r['text'] for r in parts['priors'][name]['train']]
        if sum(map(len, texts)) != audit['prior_letters']:
            raise ValueError('Unmatched model budget')
        p = CharacterPrior.fit(texts)
        path = STATE / 'priors' / f'{name}.npz'
        if path.exists():
            raise FileExistsError(path)
        p.save(path)
        prior_files[str(path.relative_to(ROOT))] = digest(path)
        model_hashes[name] = hashlib.sha256(b''.join(x.tobytes() for x in p.probabilities)).hexdigest()
        held = ''.join(r['text'] for r in parts['priors'][name]['calibration'])
        entropy[name] = p.bits(held) / len(held)
    paths = set(old['code_and_records']) | {
        'experiments/language_coverage.py', 'experiments/language_coverage_sources.py',
        'experiments/language-coverage/PROTOCOL.md', 'experiments/language-coverage/sources.json',
        'tests/test_language_coverage.py'}
    external = dict(old['external'])
    external.update({r['path']: r['sha256'] for r in audit['downloads']})
    external['artifacts/language-coverage/partitions.json'] = digest(STATE / 'partitions.json')
    external.update(prior_files)
    seal(OUT / 'freeze.json', dict(code_and_records={p: digest(ROOT / p) for p in sorted(paths)},
        external=external, prior_files=prior_files, prior_probability_sha256=model_hashes, entropy=entropy,
        settings=old['settings'], fit_cap=3600, refine_cap=1200, sweeps=200, max_evaluations=20_000_000,
        batch_size=512, workers=4, threads=2, backend='incremental', max_worker_seconds=43200,
        baseline=BASELINE, expanded=EXPANDED, models=MODELS, at=time.time(), voynich_used=False))
    print('Frozen; commit before prepare.', entropy, flush=True)


def verify():
    f = read(OUT / 'freeze.json')
    for path, expected in {**f['code_and_records'], **f['external']}.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen drift: ' + path)
    for path in [*f['code_and_records'], 'experiments/language-coverage/freeze.json']:
        data = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != digest(ROOT / path):
            raise ValueError('Uncommitted freeze: ' + path)
    return f


def prepare():
    verify()
    if (STATE / 'public').exists():
        raise FileExistsError('Challenge already prepared')
    vendor = load_vendor()
    if vendor.RESPACING != 17:
        raise ValueError('Unexpected encoder spacing')
    rng = random.SystemRandom()
    private = STATE / 'evaluator-only'
    private.mkdir(mode=0o700, exist_ok=True)
    inputs, answers, schedule = {}, [], []
    for language, pair in read(STATE / 'partitions.json')['passages'].items():
        seed = rng.randrange(2**63)
        letters = list(ALPHABET)
        random.Random(seed).shuffle(letters)
        key = dict(zip(ALPHABET, letters))
        ident = hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        seeds = [rng.randrange(2**63) for _ in range(2)]
        for stage, passage, enc_seed in zip(('fit', 'transfer'), pair, seeds):
            tokens = encrypt(vendor, passage['plaintext'], key, enc_seed)
            path = STATE / 'public' / f'{ident}-{stage}.json'
            seal(path, dict(id=ident, ciphertext=' '.join(tokens)))
            inputs[str(path.relative_to(ROOT))] = digest(path)
        answers.append(dict(id=ident, language=language, key_seed=seed, encoder_seeds=seeds,
                            passages=pair))
        schedule.append(ident)
    seal(private / 'answers.json', answers)
    seal(STATE / 'challenge.json', dict(schedule=schedule, inputs=inputs,
         answers_sha256=digest(private / 'answers.json'), freeze_sha256=digest(OUT / 'freeze.json'),
         freeze_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print('Sealed three anonymous ciphertext pairs; 24 model fits planned.', flush=True)


def worker(stage, ident, model):
    f = read(OUT / 'freeze.json')
    challenge = read(STATE / 'challenge.json')
    public = STATE / 'public' / f'{ident}-{stage}.json'
    prior_path = STATE / 'priors' / f'{model}.npz'
    for path, expected in [(public, challenge['inputs'][str(public.relative_to(ROOT))]),
                           (prior_path, f['prior_files'][str(prior_path.relative_to(ROOT))])]:
        if digest(path) != expected:
            raise ValueError('Worker input drift')
    provenance = dict(id=ident, model=model, stage=stage, freeze_sha256=digest(OUT / 'freeze.json'),
                      challenge_sha256=digest(STATE / 'challenge.json'))
    path = STATE / stage / f'{ident}-{model}.json'
    if path.exists():
        if read(path)['provenance'] != provenance:
            raise ValueError('Checkpoint provenance changed')
        return str(path)
    prior = CharacterPrior.load(prior_path)
    tokens = read(public)['ciphertext'].split()
    if stage == 'fit':
        result = previous.fit_key(tokens, prior, f)
    else:
        fit_path = STATE / 'fit' / f'{ident}-{model}.json'
        expected = read(STATE / 'sealed-keys' / f'{ident}.json')['files'][str(fit_path.relative_to(ROOT))]
        if digest(fit_path) != expected:
            raise ValueError('Key changed after sealing')
        result = transfer(tokens, read(fit_path)['result']['mapping'], prior)
    seal(path, dict(provenance=provenance, result=result))
    print(f'{stage} {ident} {model} complete' + (f" {result['seconds']:.1f}s cap={result['cap_hit']}" if stage == 'fit' else ''), flush=True)
    return str(path)


def run():
    f = verify()
    challenge = read(STATE / 'challenge.json')
    if challenge['freeze_sha256'] != digest(OUT / 'freeze.json'):
        raise ValueError('Challenge freeze mismatch')
    if (STATE / 'run.json').exists():
        raise FileExistsError('Run already ended')
    for name in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '2' if name in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS') else '1'
    completed, stop = [], None
    with ProcessPoolExecutor(max_workers=f['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for ident in challenge['schedule']:
            used = sum(read(p)['result']['seconds'] for p in (STATE / 'fit').glob('*.json'))
            missing = sum(not (STATE / 'fit' / f'{ident}-{m}.json').exists() for m in MODELS)
            if used + missing * f['fit_cap'] > f['max_worker_seconds']:
                stop = 'inconclusive_worker_budget'
                break
            futures = [pool.submit(worker, 'fit', ident, m) for m in MODELS]
            paths = [Path(job.result()) for job in futures]
            keys = dict(files={str(p.relative_to(ROOT)): digest(p) for p in paths})
            target = STATE / 'sealed-keys' / f'{ident}.json'
            if target.exists():
                if read(target) != keys:
                    raise ValueError('Key seal drift')
            else:
                seal(target, keys)
            futures = [pool.submit(worker, 'transfer', ident, m) for m in MODELS]
            for job in futures:
                job.result()
            completed.append(ident)
            if any(read(p)['result']['cap_hit'] for p in paths):
                stop = 'inconclusive_compute_cap'
                break
    seal(STATE / 'run.json', dict(completed=completed, stop_reason=stop,
        fit_worker_seconds=sum(read(p)['result']['seconds'] for p in (STATE / 'fit').glob('*.json')),
        challenge_sha256=digest(STATE / 'challenge.json')))


def evaluate():
    f = verify()
    challenge, run = read(STATE / 'challenge.json'), read(STATE / 'run.json')
    if digest(STATE / 'evaluator-only/answers.json') != challenge['answers_sha256']:
        raise ValueError('Answers changed')
    outcomes = []
    for answer in read(STATE / 'evaluator-only/answers.json'):
        ident, language = answer['id'], answer['language']
        if ident not in run['completed']:
            continue
        scores, errors, files = {}, {}, {}
        for model in MODELS:
            a_path, b_path = [STATE / stage / f'{ident}-{model}.json' for stage in ('fit', 'transfer')]
            a, b = read(a_path)['result'], read(b_path)['result']
            scores[model] = dict(fit_excess=a['bits_per_letter']-f['entropy'][model],
                transfer_excess=None if b['bits_per_letter'] is None else b['bits_per_letter']-f['entropy'][model],
                coverage=b['token_coverage'], cap_hit=a['cap_hit'], cap_hits=a['cap_hits'], seconds=a['seconds'])
            files.update({str(p.relative_to(ROOT)): digest(p) for p in (a_path, b_path)})
            if model == EXPANDED[language] or model == BASELINE.get(language):
                errors[model] = dict(fit_cer=edit_distance(a['recovered'],answer['passages'][0]['plaintext'])/5200,
                    transfer_cer=edit_distance(b['recovered'],answer['passages'][1]['plaintext'])/5200)
        outcomes.append(dict(id=ident, language=language, scores=scores, decisions=decisions(scores,language),
                             character_error=errors, files=files))
    complete = len(outcomes) == 3 and run['stop_reason'] is None
    seal(OUT / 'results.json', dict(outcomes=outcomes, **run, planned_pairs=3,
        expanded_correct=sum(r['decisions']['expanded']['accepted']==r['language'] for r in outcomes),
        omitted_rejected=sum(r['decisions']['omitted']['accepted'] is None and not r['decisions']['omitted']['inconclusive'] for r in outcomes),
        feasibility_pass=complete and all(r['decisions']['expanded']['accepted']==r['language'] and
                    r['decisions']['omitted']['accepted'] is None for r in outcomes),
        exploratory=True, voynich_used=False, freeze_sha256=digest(OUT/'freeze.json')))
    print(json.dumps(read(OUT / 'results.json'),indent=2), flush=True)


def replay():
    f = verify()
    n = 0
    for row in read(OUT / 'results.json')['outcomes']:
        for path, expected in row['files'].items():
            if digest(ROOT / path) != expected:
                raise ValueError('Result drift')
        ident = row['id']
        scores = {}
        for model in MODELS:
            path = STATE / 'fit' / f'{ident}-{model}.json'
            keys = read(STATE / 'sealed-keys' / f'{ident}.json')['files']
            if digest(path) != keys[str(path.relative_to(ROOT))]:
                raise ValueError('Key seal drift')
            a = read(path)['result']
            p = CharacterPrior.load(STATE / 'priors' / f'{model}.npz')
            actual = transfer(read(STATE / 'public' / f'{ident}-transfer.json')['ciphertext'].split(), a['mapping'], p)
            expected = read(STATE / 'transfer' / f'{ident}-{model}.json')['result']
            if actual != expected or abs(p.bits(a['recovered']) / len(a['recovered']) - a['bits_per_letter']) > 1e-10:
                raise ValueError('Score replay mismatch')
            scores[model] = dict(fit_excess=a['bits_per_letter']-f['entropy'][model],
                transfer_excess=None if actual['bits_per_letter'] is None else actual['bits_per_letter']-f['entropy'][model],
                coverage=actual['token_coverage'], cap_hit=a['cap_hit'], cap_hits=a['cap_hits'], seconds=a['seconds'])
            n += 1
        if scores != row['scores'] or decisions(scores, row['language']) != row['decisions']:
            raise ValueError('Decision drift')
    print(json.dumps(dict(replayed_transfers=n, decisions_reproduce=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze','verify','prepare','run','evaluate','replay'])
    globals()[parser.parse_args().command]()
