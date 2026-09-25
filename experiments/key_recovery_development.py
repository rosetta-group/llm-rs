"""Development only: whole-token admission after the frozen fit, on the five released language cases.

Every candidate model is refitted on every released fit passage twice from the same frozen
stages: B is the frozen `rejection_followups_v2.fit_key` result, A adds leave-one-out
whole-token admission and two context-reparse rounds. Keys are fixed before transfer. The
decision rule and thresholds are the frozen `decide_transfer`. Answers are read only by
`evaluate`. No reserved Voynich text, no fresh passages.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from experiments.joint_development import majority_key, role_units
from experiments import rejection_transfer_v2 as previous
from voynich.context_reparse import reparse
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer
from voynich.variable_units_bounded import refine
from voynich.whole_admission import admit_wholes, apply_wholes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/key-recovery-development'
STATE = ROOT / 'artifacts/key-recovery-development'
FREEZE = ROOT / 'experiments/language-expansion/freeze.json'
PRIORS = ROOT / 'artifacts/language-expansion/priors'
EXPERIMENTS = ('language-coverage', 'language-expansion')
TRUE_MODEL = dict(catalan='catalan', german='german_broad', latin='latin_broad', czech='czech', occitan='occitan',
                  english='english', italian='italian')
ADMISSION_ROUNDS, REPARSE_ROUNDS, REPARSE_WIDTH = 3, 2, 128
read, seal = previous.read, previous.seal


def released_cases():
    """(experiment, id, language) from the published results; answers are opened only by `evaluate`."""
    return [(e, row['id'], row['language']) for e in EXPERIMENTS
            for row in read(ROOT / 'experiments' / e / 'results.json')['outcomes']]


def released_controls():
    """(experiment, id, kind) for the graded rejection-screen inputs only; unscored prepared blocks are excluded."""
    graded = [('rejection-transfer-v2', row['id'], row['kind'])
              for row in read(ROOT / 'experiments/rejection-transfer-v2/results.json')['outcomes'] if row['kind'] != 'absent']
    copies = [('rejection-followups', row['id'], 'frequency_copy')
              for row in read(ROOT / 'experiments/rejection-followups/control-results.json')['outcomes']]
    return graded + copies


def refit(units, prior, key, seed, f, cap):
    s = f['settings']
    return refine(units, prior, key, (), seed=seed, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                  cap=cap, sweeps=f['sweeps'], max_evaluations=f['max_evaluations'],
                  batch_size=f['batch_size'], backend=f['backend'])


def fit_both(tokens, prior, f):
    """Frozen stages once; B is their result, A continues from it."""
    s, cap = f['settings'], f['fit_cap']
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = previous.joint_em(tokens, prior, **em)
    second = previous.prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = previous.repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'],
                               passes=s['repair_passes'], usage_floor=s['repair_usage_floor'], **em)
    segmentation = [tuple(p) for p in repaired['segmentation']]
    units = role_units(segmentation)
    ref = refit(units, prior, majority_key(units, repaired['recovered']), 1,
                f, min(f['refine_cap'], max(30., cap - (time.monotonic() - started))))
    hits = dict(first=first['cap_hit'], second=second['cap_hit'], repair=repaired['cap_hit'], refine=ref['cap_hit'])
    base_seconds = time.monotonic() - started
    B = dict(recovered=ref['recovered'], mapping=ref['mapping'], cap_hit=any(hits.values()) or base_seconds > cap,
             seconds=base_seconds, bits_per_letter=prior.bits(ref['recovered']) / len(ref['recovered']))
    rounds = []
    for r in range(ADMISSION_ROUNDS):
        admitted, record = admit_wholes(tokens, segmentation, ref['mapping'], prior)
        rounds.append(record)
        if not admitted:
            break
        segmentation = apply_wholes(tokens, segmentation, admitted)
        units = role_units(segmentation)
        key = {u: ref['mapping'].get(u, 'a') for u in set(units)}
        key.update({'u:' + t: y for t, y in admitted.items()})
        ref = refit(units, prior, key, 10 + r, f, f['refine_cap'])
        hits[f'admission_{r}'] = ref['cap_hit']
    for r in range(REPARSE_ROUNDS):
        parsed = reparse(tokens, ref['mapping'], prior, width=REPARSE_WIDTH)
        segmentation = [tuple(p) for p in parsed['segmentation']]
        units = role_units(segmentation)
        ref = refit(units, prior, majority_key(units, parsed['recovered']), 20 + r, f, f['refine_cap'])
        hits[f'reparse_{r}'] = ref['cap_hit']
    seconds = time.monotonic() - started
    A = dict(recovered=ref['recovered'], mapping=ref['mapping'], cap_hit=any(hits.values()) or seconds > 2 * cap,
             seconds=seconds, bits_per_letter=prior.bits(ref['recovered']) / len(ref['recovered']), admission=rounds)
    return dict(B=B, A=A, cap_hits=hits)


def worker(job):
    experiment, ident, model = job
    path = STATE / 'fit' / f'{ident}-{model}.json'
    if path.exists():
        return str(path)
    f = read(FREEZE)
    prior = CharacterPrior.load(PRIORS / f'{model}.npz')
    if digest(PRIORS / f'{model}.npz') != f['prior_files'][str((PRIORS / f'{model}.npz').relative_to(ROOT))]:
        raise ValueError('Prior changed')
    public = ROOT / 'artifacts' / experiment / 'public'
    fit_tokens = read(public / f'{ident}-fit.json')['ciphertext'].split()
    held_tokens = read(public / f'{ident}-transfer.json')['ciphertext'].split()
    result = fit_both(fit_tokens, prior, f)
    # Keys are final before the transfer passage is decoded.
    for arm in ('B', 'A'):
        result[arm]['transfer'] = transfer(held_tokens, result[arm]['mapping'], prior)
    seal(path, dict(experiment=experiment, id=ident, model=model, result=result, freeze_sha256=digest(FREEZE)))
    print(f'{ident} {model} B {result["B"]["seconds"]:.0f}s A {result["A"]["seconds"]:.0f}s', flush=True)
    return str(path)


def run(workers, controls=False):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    models = read(FREEZE)['models']
    jobs = [(e, i, m) for e, i, _ in (released_controls() if controls else released_cases()) for m in models]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(worker, jobs))
    print('Fitted', len(jobs), 'case-model pairs.', flush=True)


def true_texts(experiment, ident):
    answer = [a for a in read(ROOT / 'artifacts' / experiment / 'evaluator-only/answers.json') if a['id'] == ident][0]
    return answer['language'], [p['plaintext'] for p in answer['passages']]


def evaluate():
    f = read(FREEZE)
    languages = f['expanded']
    rows = []
    for experiment, ident, language in released_cases():
        truth, (fit_text, held_text) = true_texts(experiment, ident)
        if truth != language:
            raise ValueError('Published language differs from answer')
        fits = {m: read(STATE / 'fit' / f'{ident}-{m}.json')['result'] for m in f['models']}
        row = dict(language=truth, id=ident, experiment=experiment)
        for arm in ('B', 'A'):
            scores = {}
            for label, model in languages.items():
                r = fits[model][arm]
                scores[label] = dict(fit_excess=r['bits_per_letter'] - f['entropy'][model],
                                     transfer_excess=None if r['transfer']['bits_per_letter'] is None
                                     else r['transfer']['bits_per_letter'] - f['entropy'][model],
                                     coverage=r['transfer']['token_coverage'], cap_hit=r['cap_hit'])
            own = fits[TRUE_MODEL[truth]][arm]
            row[arm] = dict(scores=scores, full=decide_transfer(scores),
                            omitted=decide_transfer(scores, [l for l in languages if l != truth]),
                            fit_cer=edit_distance(own['recovered'], fit_text) / len(fit_text),
                            transfer_cer=edit_distance(own['transfer']['recovered'], held_text) / len(held_text))
        rows.append(row)
    summary = {arm: dict(accepted_correct=sum(r[arm]['full']['accepted'] == r['language'] for r in rows),
                         accepted_wrong=sum(r[arm]['full']['accepted'] not in (None, r['language']) for r in rows),
                         omitted_accepted=sum(r[arm]['omitted']['accepted'] is not None for r in rows),
                         inconclusive=sum(r[arm]['full']['inconclusive'] or r[arm]['omitted']['inconclusive'] for r in rows))
               for arm in ('B', 'A')}
    seal(OUT / 'results.json', dict(rows=rows, summary=summary, development_only=True, voynich_used=False,
                                    freeze_sha256=digest(FREEZE), at=time.time()))
    for r in rows:
        print(r['language'], *(f"{arm}: accepted={r[arm]['full']['accepted']} reasons={r[arm]['full']['reasons']} "
                                f"omitted={r[arm]['omitted']['accepted']} CER {100*r[arm]['fit_cer']:.1f}/{100*r[arm]['transfer_cer']:.1f}"
                                for arm in ('B', 'A')), sep='\n  ')
    print(json.dumps(summary, indent=1))


def scores_for(fits, arm, f):
    out = {}
    for label, model in f['expanded'].items():
        r = fits[model][arm]
        out[label] = dict(fit_excess=r['bits_per_letter'] - f['entropy'][model],
                          transfer_excess=None if r['transfer']['bits_per_letter'] is None
                          else r['transfer']['bits_per_letter'] - f['entropy'][model],
                          coverage=r['transfer']['token_coverage'], cap_hit=r['cap_hit'])
    return out


def evaluate_controls():
    f = read(FREEZE)
    answers = read(ROOT / 'artifacts/rejection-transfer-v2/released-answers.json')
    rows = []
    for experiment, ident, kind in released_controls():
        fits = {m: read(STATE / 'fit' / f'{ident}-{m}.json')['result'] for m in f['models']}
        language = answers[ident]['language'] if ident in answers else None
        row = dict(id=ident, kind=kind, language=language, experiment=experiment)
        for arm in ('B', 'A'):
            scores = scores_for(fits, arm, f)
            row[arm] = dict(scores=scores, full=decide_transfer(scores),
                            seconds=sum(fits[m][arm]['seconds'] for m in f['models']))
            if kind == 'positive':
                row[arm]['omitted'] = decide_transfer(scores, [l for l in f['expanded'] if l != language])
                own = fits[TRUE_MODEL[language]][arm]
                texts = [p['plaintext'].replace(' ', '') for p in answers[ident]['passages']]
                row[arm]['fit_cer'] = edit_distance(own['recovered'], texts[0]) / len(texts[0])
                row[arm]['transfer_cer'] = edit_distance(own['transfer']['recovered'], texts[1]) / len(texts[1])
        rows.append(row)
    summary = {arm: dict(positives_correct=sum(r[arm]['full']['accepted'] == r['language'] for r in rows if r['kind'] == 'positive'),
                         positives_wrong=sum(r[arm]['full']['accepted'] not in (None, r['language']) for r in rows if r['kind'] == 'positive'),
                         omitted_accepted=sum(r[arm]['omitted']['accepted'] is not None for r in rows if r['kind'] == 'positive'),
                         negatives_accepted=sum(r[arm]['full']['accepted'] is not None for r in rows if r['kind'] != 'positive'),
                         inconclusive=sum(r[arm]['full']['inconclusive'] for r in rows),
                         mean_fit_seconds=sum(r[arm]['seconds'] for r in rows) / (len(rows) * len(f['models'])))
               for arm in ('B', 'A')}
    seal(OUT / 'controls-results.json', dict(rows=rows, summary=summary, development_only=True, voynich_used=False,
                                             freeze_sha256=digest(FREEZE), at=time.time()))
    for r in rows:
        print(r['kind'], r['language'], *(f"{arm}: accepted={r[arm]['full']['accepted']} reasons={r[arm]['full']['reasons']}" for arm in ('B', 'A')))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'evaluate', 'run_controls', 'evaluate_controls'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.command.startswith('run'):
        run(args.workers, controls=args.command == 'run_controls')
    else:
        evaluate() if args.command == 'evaluate' else evaluate_controls()
