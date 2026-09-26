"""Second fresh confirmation: broadened Latin, German and Catalan priors; 24 sealed blocks, one run.

python -m experiments.key_recovery_confirmation_v2 freeze | verify | prepare | run | evaluate

See experiments/key-recovery-confirmation-v2/PROTOCOL.md. The solver sees only ciphertext and the
eight frozen priors. Answers are written once by `prepare` and read only by `evaluate`.
"""
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

from experiments.joint_development import load_vendor
from experiments import key_recovery_v2_sources as sources
from experiments import key_recovery_v2_development as v2
from experiments import key_recovery_development as development
from experiments.rejection_transfer_v2 import encrypt, read, seal
from voynich.data import digest
from voynich.decipher import ALPHABET, edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer, frequency_copy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/key-recovery-confirmation-v2'
STATE = ROOT / 'artifacts/key-recovery-confirmation-v2'
PRIOR_FREEZE = ROOT / 'experiments/language-expansion/freeze.json'
KINDS = ('positive', 'shuffle', 'frequency_copy')   # copy-mutate always hits the frozen work limit
BLOCKS_PER_LANGUAGE = 3
WORKERS, THREADS = 5, 2
MAX_WORKER_SECONDS = 150 * 3600
RESERVE_PER_FIT = 900          # seconds reserved per unfitted input-model pair before a block starts
CODE = ('experiments/key_recovery_confirmation_v2.py', 'experiments/key_recovery_v2_sources.py',
        'experiments/key_recovery_v2_development.py', 'experiments/key-recovery-v2-development/priors.json',
        'experiments/key_recovery_development.py', 'experiments/rejection_transfer_v2.py',
        'experiments/joint_development.py', 'voynich/whole_admission.py', 'voynich/piece_admission.py',
        'voynich/context_reparse.py', 'voynich/rejection.py', 'voynich/rejection_development.py',
        'voynich/variable_units.py', 'voynich/variable_units_bounded.py', 'voynich/joint_segments.py',
        'voynich/joint_segments_v2.py', 'voynich/lexicon_repair.py', 'voynich/description_length.py',
        'experiments/key-recovery-confirmation-v2/PROTOCOL.md', 'experiments/key-recovery-confirmation-v2/sources.json',
        'tests/test_whole_admission.py', 'tests/test_key_recovery_v2_sources.py')


def prior_path(model):
    return (v2.STATE / 'priors' if model in v2.NEW.values() else ROOT / 'artifacts/language-expansion/priors') / f'{model}.npz'


def candidates():
    """(label -> model, model -> calibration entropy): three broadened priors, five unchanged."""
    labels = v2.labels()
    entropy = dict(read(PRIOR_FREEZE)['entropy'], **{m: r['entropy'] for m, r in read(v2.OUT / 'priors.json')['models'].items()})
    return labels, {m: entropy[m] for m in labels.values()}


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Already frozen')
    manifest = read(OUT / 'sources.json')
    if digest(STATE / 'partitions.json') != manifest['partitions_sha256']:
        raise ValueError('Partition drift')
    seal(OUT / 'freeze.json', dict(
        code={p: digest(ROOT / p) for p in CODE}, prior_freeze_sha256=digest(PRIOR_FREEZE),
        prior_files={str(prior_path(m).relative_to(ROOT)): digest(prior_path(m)) for m in candidates()[0].values()},
        labels=candidates()[0], entropy=candidates()[1], partitions_sha256=manifest['partitions_sha256'],
        languages=sources.LANGUAGES, blocks_per_language=BLOCKS_PER_LANGUAGE, kinds=KINDS,
        workers=WORKERS, threads=THREADS, max_worker_seconds=MAX_WORKER_SECONDS, reserve_per_fit=RESERVE_PER_FIT,
        a_cap=2 * read(PRIOR_FREEZE)['fit_cap'], b_cap=read(PRIOR_FREEZE)['fit_cap'], voynich_used=False, at=time.time()))
    print('Frozen. Commit before prepare.', flush=True)


def verify():
    f = read(OUT / 'freeze.json')
    for path, expected in {**f['code'], **f['prior_files']}.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen drift: ' + path)
    if digest(PRIOR_FREEZE) != f['prior_freeze_sha256'] or digest(STATE / 'partitions.json') != f['partitions_sha256']:
        raise ValueError('Prior freeze or partition drift')
    for path in [*f['code'], 'experiments/key-recovery-confirmation/freeze.json']:
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest(ROOT / path):
            raise ValueError('Uncommitted freeze: ' + path)
    return f


def prepare():
    f = verify()
    if (STATE / 'challenge.json').exists() or (STATE / 'public').exists():
        raise FileExistsError('Already prepared, or incomplete preparation needs investigation')
    passages = read(STATE / 'partitions.json')['passages']
    vendor = load_vendor()
    if vendor.RESPACING != 17:
        raise ValueError('Unexpected encoder spacing')
    rng = random.SystemRandom()
    private = STATE / 'evaluator-only'
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    answers, schedule, inputs = {}, [], {}
    # Pairs are fixed by passage index (0,1), (2,3), (4,5); which passage is fitted is random.
    for round_ in range(f['blocks_per_language']):
        for language in f['languages']:
            pair = passages[language][2 * round_:2 * round_ + 2]
            if rng.random() < .5:
                pair = pair[::-1]
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
            for kind, tokens_pair in zip(KINDS, (positive, shuffled, frequency)):
                ident = hashlib.sha256(str(rng.randrange(2 ** 128)).encode()).hexdigest()[:20]
                ids.append(ident)
                for role, tokens in zip(('fit', 'transfer'), tokens_pair):
                    path = STATE / 'public' / f'{ident}-{role}.json'
                    seal(path, dict(id=ident, ciphertext=' '.join(tokens)))
                    inputs[str(path.relative_to(ROOT))] = digest(path)
                answers[ident] = dict(language=language, kind=kind, block=len(schedule), passages=pair,
                                      key_seed=key_seed, seeds=seeds, key=key)
            schedule.append(dict(block=len(schedule), inputs=ids))
    seal(private / 'answers.json', answers)
    seal(STATE / 'challenge.json', dict(schedule=[b['inputs'] for b in schedule], inputs=inputs,
         answers_sha256=digest(private / 'answers.json'), freeze_sha256=digest(OUT / 'freeze.json'),
         freeze_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), at=time.time()))
    print(f'Sealed {len(schedule)} blocks, {len(inputs) // 2} anonymous inputs.', flush=True)


def worker(stage, ident, model):
    f = read(OUT / 'freeze.json')
    challenge = read(STATE / 'challenge.json')
    public = STATE / 'public' / f'{ident}-{stage}.json'
    path_ = prior_path(model)
    if digest(public) != challenge['inputs'][str(public.relative_to(ROOT))] or \
            digest(path_) != f['prior_files'][str(path_.relative_to(ROOT))]:
        raise ValueError('Worker input drift')
    provenance = dict(id=ident, model=model, stage=stage, freeze_sha256=digest(OUT / 'freeze.json'),
                      challenge_sha256=digest(STATE / 'challenge.json'))
    path = STATE / stage / f'{ident}-{model}.json'
    if path.exists():
        if read(path)['provenance'] != provenance:
            raise ValueError('Checkpoint provenance changed')
        return str(path)
    prior = CharacterPrior.load(path_)
    tokens = read(public)['ciphertext'].split()
    if stage == 'fit':
        result = development.fit_both(tokens, prior, read(PRIOR_FREEZE))
    else:
        fit_path = STATE / 'fit' / f'{ident}-{model}.json'
        if digest(fit_path) != read(STATE / 'sealed-keys' / f'{ident}.json')['files'][str(fit_path.relative_to(ROOT))]:
            raise ValueError('Key changed after sealing')
        fitted = read(fit_path)['result']
        result = {arm: transfer(tokens, fitted[arm]['mapping'], prior) for arm in ('B', 'A')}
    seal(path, dict(provenance=provenance, result=result))
    return str(path)


def used_seconds():
    return sum(read(p)['result']['A']['seconds'] for p in (STATE / 'fit').glob('*.json'))


def run():
    f = verify()
    challenge = read(STATE / 'challenge.json')
    if challenge['freeze_sha256'] != digest(OUT / 'freeze.json'):
        raise ValueError('Challenge freeze mismatch')
    if (STATE / 'run.json').exists():
        raise FileExistsError('Run already ended')
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(f['threads']) if name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS') else '1'
    models = list(f['labels'].values())
    completed, stop = [], None
    with ProcessPoolExecutor(max_workers=f['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for block in challenge['schedule']:
            missing = sum(not (STATE / 'fit' / f'{i}-{m}.json').exists() for i in block for m in models)
            if (STATE / 'fit').exists() and used_seconds() + missing * f['reserve_per_fit'] > f['max_worker_seconds']:
                stop = 'inconclusive_worker_budget'
                break
            jobs = {i: [pool.submit(worker, 'fit', i, m) for m in models] for i in block}
            for ident, futures in jobs.items():
                paths = [Path(job.result()) for job in futures]
                keys = dict(files={str(p.relative_to(ROOT)): digest(p) for p in paths})
                target = STATE / 'sealed-keys' / f'{ident}.json'
                if target.exists():
                    if read(target) != keys:
                        raise ValueError('Key seal drift')
                else:
                    seal(target, keys)
            for job in [pool.submit(worker, 'transfer', i, m) for i in block for m in models]:
                job.result()
            completed.append(block)
            print(f'block {len(completed)}/{len(challenge["schedule"])} done; {used_seconds() / 3600:.1f} worker-hours', flush=True)
    seal(STATE / 'run.json', dict(completed=completed, stop_reason=stop, fit_worker_seconds=used_seconds(),
                                  challenge_sha256=digest(STATE / 'challenge.json')))


def scores(ident, arm, models, entropy, labels):
    out = {}
    for label, model in labels.items():
        fit = read(STATE / 'fit' / f'{ident}-{model}.json')['result'][arm]
        held = read(STATE / 'transfer' / f'{ident}-{model}.json')['result'][arm]
        out[label] = dict(fit_excess=fit['bits_per_letter'] - entropy[model],
                          transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - entropy[model],
                          coverage=held['token_coverage'], cap_hit=fit['cap_hit'])
    return out


def evaluate():
    verify()
    challenge, run_record = read(STATE / 'challenge.json'), read(STATE / 'run.json')
    if digest(STATE / 'evaluator-only/answers.json') != challenge['answers_sha256']:
        raise ValueError('Answers changed')
    f = read(OUT / 'freeze.json')
    labels, entropy = f['labels'], f['entropy']
    answers = read(STATE / 'evaluator-only/answers.json')
    rows = []
    for block in run_record['completed']:
        for ident in block:
            answer = answers[ident]
            row = dict(id=ident, block=answer['block'], language=answer['language'], kind=answer['kind'],
                       works=[p['document'] for p in answer['passages']])
            for arm in ('B', 'A'):
                s = scores(ident, arm, list(labels.values()), entropy, labels)
                row[arm] = dict(scores=s, full=decide_transfer(s))
                if answer['kind'] == 'positive':
                    row[arm]['omitted'] = decide_transfer(s, [l for l in labels if l != answer['language']])
                    model = labels[answer['language']]
                    fit = read(STATE / 'fit' / f'{ident}-{model}.json')['result'][arm]
                    held = read(STATE / 'transfer' / f'{ident}-{model}.json')['result'][arm]
                    texts = [p['plaintext'] for p in answer['passages']]
                    row[arm]['fit_cer'] = edit_distance(fit['recovered'], texts[0]) / len(texts[0])
                    row[arm]['transfer_cer'] = edit_distance(held['recovered'], texts[1]) / len(texts[1])
            rows.append(row)
    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] != 'positive']
    summary = {}
    for arm in ('B', 'A'):
        summary[arm] = dict(
            positives=len(positives),
            correct=sum(r[arm]['full']['accepted'] == r['language'] for r in positives),
            wrong_language=sum(r[arm]['full']['accepted'] not in (None, r['language']) for r in positives),
            omitted_accepted=sum(r[arm]['omitted']['accepted'] is not None for r in positives),
            negatives=len(negatives),
            negatives_accepted=sum(r[arm]['full']['accepted'] is not None for r in negatives),
            inconclusive=sum(r[arm]['full']['inconclusive'] for r in rows),
            by_language={l: sum(r[arm]['full']['accepted'] == l for r in positives if r['language'] == l) for l in labels})
    a, b = summary['A'], summary['B']
    safety = a['wrong_language'] == 0 and a['omitted_accepted'] == 0 and a['negatives_accepted'] == 0
    complete = run_record['stop_reason'] is None and a['positives'] == 24 and a['inconclusive'] == 0
    seal(OUT / 'results.json', dict(rows=rows, summary=summary, safety=safety, sensitivity_pass=a['correct'] >= 16,
         paired_improvement=a['correct'] > b['correct'], complete=complete,
         confirmation_pass=complete and safety and a['correct'] >= 16 and a['correct'] > b['correct'],
         stop_reason=run_record['stop_reason'], fit_worker_seconds=run_record['fit_worker_seconds'],
         freeze_sha256=digest(OUT / 'freeze.json'), voynich_used=False))
    print(json.dumps(read(OUT / 'results.json')['summary'], indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'verify', 'prepare', 'run', 'evaluate'])
    globals()[parser.parse_args().command]()
