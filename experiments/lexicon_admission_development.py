"""Development only: decoder C, joint whole and half piece admission fed back through joint EM.

python -m experiments.lexicon_admission_development quick | quick_report | run | evaluate

C starts from decoder A's final state. Each round proposes rare whole tokens and rare half pieces
by the leave-one-out context tests, adds them to the lexicon, reruns joint EM from scratch on the
enlarged lexicon, then refines and reparses as A does. `quick` runs the 24 released positives of
the second confirmation under their true-language prior only; answers are used to score cost.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

from experiments import key_recovery_confirmation_v2 as confirmation
from experiments import key_recovery_development as development
from experiments import recovery_oracles as oracles
from experiments import rejection_transfer_v2 as previous
from experiments.joint_development import load_vendor, majority_key, role_units
from experiments.rejection_transfer_v2 import read, seal
from voynich.context_reparse import reparse
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.half_admission import admit_halves
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer
from voynich.rejection_v3 import rescore
from voynich.whole_admission import admit_wholes, apply_wholes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/lexicon-admission-development'
STATE = ROOT / 'artifacts/lexicon-admission-development'
C_ROUNDS = 2


def fit_ac(tokens, prior, f):
    """A exactly as `development.fit_both`, keeping its lexicon and final segmentation; then C."""
    s, cap = f['settings'], f['fit_cap']
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = previous.joint_em(tokens, prior, **em)
    second = previous.prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = previous.repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'],
                               passes=s['repair_passes'], usage_floor=s['repair_usage_floor'], **em)
    segmentation = [tuple(p) for p in repaired['segmentation']]
    units = role_units(segmentation)
    ref = development.refit(units, prior, majority_key(units, repaired['recovered']), 1,
                            f, min(f['refine_cap'], max(30., cap - (time.monotonic() - started))))
    hits = dict(first=first['cap_hit'], second=second['cap_hit'], repair=repaired['cap_hit'], refine=ref['cap_hit'])
    lexicon = set(repaired['candidate_pieces'])
    for r in range(development.ADMISSION_ROUNDS):
        admitted, _ = admit_wholes(tokens, segmentation, ref['mapping'], prior)
        if not admitted:
            break
        lexicon |= set(admitted)
        segmentation = apply_wholes(tokens, segmentation, admitted)
        units = role_units(segmentation)
        key = {u: ref['mapping'].get(u, 'a') for u in set(units)}
        key.update({'u:' + t: y for t, y in admitted.items()})
        ref = development.refit(units, prior, key, 10 + r, f, f['refine_cap'])
        hits[f'admission_{r}'] = ref['cap_hit']
    for r in range(development.REPARSE_ROUNDS):
        parsed = reparse(tokens, ref['mapping'], prior, width=development.REPARSE_WIDTH)
        segmentation = [tuple(p) for p in parsed['segmentation']]
        units = role_units(segmentation)
        ref = development.refit(units, prior, majority_key(units, parsed['recovered']), 20 + r, f, f['refine_cap'])
        hits[f'reparse_{r}'] = ref['cap_hit']
    a_seconds = time.monotonic() - started
    A = dict(recovered=ref['recovered'], mapping=ref['mapping'], seconds=a_seconds,
             cap_hit=any(hits.values()) or a_seconds > 2 * cap, bits_per_letter=prior.bits(ref['recovered']) / len(ref['recovered']))
    lexicon |= {u[2:] for u in ref['mapping']}
    rounds = []
    for r in range(C_ROUNDS):
        wholes, whole_record = admit_wholes(tokens, segmentation, ref['mapping'], prior)
        halves, half_record = admit_halves(tokens, segmentation, ref['mapping'], prior)
        new = (set(wholes) | {u[2:] for u in halves}) - lexicon
        rounds.append(dict(wholes=whole_record, halves=half_record, new_pieces=len(new)))
        if not new:
            break
        lexicon |= new
        result = previous.joint_em(tokens, prior, pieces=lexicon, **em)
        hits[f'c_em_{r}'] = result['cap_hit']
        segmentation = [tuple(p) for p in result['segmentation']]
        units = role_units(segmentation)
        ref = development.refit(units, prior, majority_key(units, result['recovered']), 30 + 3 * r, f, f['refine_cap'])
        hits[f'c_refine_{r}'] = ref['cap_hit']
        for k in range(development.REPARSE_ROUNDS):
            parsed = reparse(tokens, ref['mapping'], prior, width=development.REPARSE_WIDTH)
            segmentation = [tuple(p) for p in parsed['segmentation']]
            units = role_units(segmentation)
            ref = development.refit(units, prior, majority_key(units, parsed['recovered']), 31 + 3 * r + k, f, f['refine_cap'])
            hits[f'c_reparse_{r}_{k}'] = ref['cap_hit']
    seconds = time.monotonic() - started
    C = dict(recovered=ref['recovered'], mapping=ref['mapping'], seconds=seconds, rounds=rounds,
             cap_hit=any(hits.values()) or seconds > 3 * cap, bits_per_letter=prior.bits(ref['recovered']) / len(ref['recovered']))
    return dict(A=A, C=C, cap_hits=hits)


def quick_case(ident):
    path = STATE / 'quick' / f'{ident}.json'
    if path.exists():
        return str(path)
    freeze = read(confirmation.OUT / 'freeze.json')
    answer = read(confirmation.STATE / 'evaluator-only/answers.json')[ident]
    model = freeze['labels'][answer['language']]
    prior, entropy = CharacterPrior.load(confirmation.prior_path(model)), freeze['entropy'][model]
    vendor = load_vendor()
    fit_tokens, _, fit_text = oracles.traced(vendor, answer['passages'][0]['plaintext'], answer['key'], answer['seeds'][0])
    held_tokens, _, held_text = oracles.traced(vendor, answer['passages'][1]['plaintext'], answer['key'], answer['seeds'][1])
    result = fit_ac(fit_tokens, prior, read(confirmation.PRIOR_FREEZE))
    archived = read(confirmation.STATE / 'fit' / f'{ident}-{model}.json')['result']['A']
    if result['A']['mapping'] != archived['mapping']:
        raise ValueError('A did not reproduce the archived key')
    true_transfer = prior.bits(held_text) / len(held_text) - entropy
    rows = {}
    for arm in ('A', 'C'):
        r = result[arm]
        held = transfer(held_tokens, r['mapping'], prior)
        rows[arm] = dict(transfer_excess=held['bits_per_letter'] - entropy,
                         transfer_cost=held['bits_per_letter'] - entropy - true_transfer,
                         fit_cer=edit_distance(r['recovered'], fit_text) / len(fit_text),
                         transfer_cer=edit_distance(held['recovered'], held_text) / len(held_text),
                         coverage=held['token_coverage'], seconds=r['seconds'], cap_hit=r['cap_hit'])
    seal(path, dict(id=ident, block=answer['block'], language=answer['language'], rows=rows,
                    c_rounds=result['C']['rounds'], development_only=True))
    print(f"{answer['block']} {answer['language']}: cost A {rows['A']['transfer_cost']:.3f} C {rows['C']['transfer_cost']:.3f}", flush=True)
    return str(path)


def quick(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    answers = read(confirmation.STATE / 'evaluator-only/answers.json')
    ids = sorted((i for i, a in answers.items() if a['kind'] == 'positive'), key=lambda i: answers[i]['block'])
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(quick_case, ids))


def quick_report():
    rows = [read(p) for p in sorted((STATE / 'quick').glob('*.json'))]
    summary = {arm: {k: statistics.median(r['rows'][arm][k] for r in rows)
                     for k in ('transfer_cost', 'transfer_excess', 'transfer_cer', 'fit_cer', 'seconds')} for arm in ('A', 'C')}
    summary['C_better_cost'] = sum(r['rows']['C']['transfer_cost'] < r['rows']['A']['transfer_cost'] for r in rows)
    summary['A_over_ceiling'] = sum(r['rows']['A']['transfer_excess'] > .5 for r in rows)
    summary['C_over_ceiling'] = sum(r['rows']['C']['transfer_excess'] > .5 for r in rows)
    summary['caps'] = sum(r['rows']['C']['cap_hit'] for r in rows)
    seal(OUT / 'quick-results.json', dict(rows=rows, summary=summary, cases=len(rows), development_only=True, at=time.time()))
    print(json.dumps(summary, indent=1))


def inputs():
    return [i for block in read(confirmation.STATE / 'challenge.json')['schedule'] for i in block]


def full_case(job):
    """All eight priors on every released v2 input: A (must match the archive) and C, fixed keys, transfer."""
    ident, model = job
    path = STATE / 'full' / f'{ident}-{model}.json'
    if path.exists():
        return str(path)
    prior = CharacterPrior.load(confirmation.prior_path(model))
    public = confirmation.STATE / 'public'
    result = fit_ac(read(public / f'{ident}-fit.json')['ciphertext'].split(), prior, read(confirmation.PRIOR_FREEZE))
    archived = read(confirmation.STATE / 'fit' / f'{ident}-{model}.json')['result']['A']['mapping']
    held = read(public / f'{ident}-transfer.json')['ciphertext'].split()
    for arm in ('A', 'C'):
        result[arm]['transfer'] = transfer(held, result[arm]['mapping'], prior)
    seal(path, dict(id=ident, model=model, result=result, a_matches_archive=result['A']['mapping'] == archived))
    return str(path)


def run(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    models = list(read(confirmation.OUT / 'freeze.json')['labels'].values())
    jobs = [(i, m) for i in inputs() for m in models]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, _ in enumerate(pool.map(full_case, jobs), 1):
            if done % 24 == 0:
                print(f'{done}/{len(jobs)} fits', flush=True)
    print('Fitted', len(jobs), 'input-model pairs.', flush=True)


def evaluate(ceilings=(.45, .5)):
    freeze = read(confirmation.OUT / 'freeze.json')
    labels, entropy = freeze['labels'], freeze['entropy']
    priors = {m: CharacterPrior.load(confirmation.prior_path(m)) for m in labels.values()}
    answers = read(confirmation.STATE / 'evaluator-only/answers.json')
    rows, mismatches = [], 0
    for ident in inputs():
        answer = answers[ident]
        fits = {m: read(STATE / 'full' / f'{ident}-{m}.json') for m in labels.values()}
        mismatches += sum(not f['a_matches_archive'] for f in fits.values())
        row = dict(id=ident, block=answer['block'], language=answer['language'], kind=answer['kind'])
        for arm in ('A', 'C'):
            for scoring in ('per_run', 'one_code'):
                scores = {}
                for label, model in labels.items():
                    r = fits[model]['result'][arm]
                    held = r['transfer'] if scoring == 'per_run' else rescore(r['transfer'], priors[model])
                    scores[label] = dict(fit_excess=r['bits_per_letter'] - entropy[model],
                                         transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - entropy[model],
                                         coverage=held['token_coverage'], cap_hit=r['cap_hit'])
                for ceiling in ceilings:
                    decision = dict(full=decide_transfer(scores, transfer_ceiling=ceiling))
                    if answer['kind'] == 'positive':
                        decision['omitted'] = decide_transfer(scores, [l for l in labels if l != answer['language']], ceiling)
                    row[f'{arm}_{scoring}_{ceiling}'] = decision
        rows.append(row)
    positives = [r for r in rows if r['kind'] == 'positive']
    negatives = [r for r in rows if r['kind'] != 'positive']
    summary = {}
    for key in [k for k in rows[0] if k[:2] in ('A_', 'C_')]:
        summary[key] = dict(correct=sum(r[key]['full']['accepted'] == r['language'] for r in positives),
                            wrong=sum(r[key]['full']['accepted'] not in (None, r['language']) for r in positives),
                            omitted=sum(r[key]['omitted']['accepted'] is not None for r in positives),
                            negatives=sum(r[key]['full']['accepted'] is not None for r in negatives),
                            inconclusive=sum(r[key]['full']['inconclusive'] for r in rows))
    seal(OUT / 'full-results.json', dict(rows=rows, summary=summary, a_archive_mismatches=mismatches,
                                         development_only=True, released_inputs=True, at=time.time()))
    print('A archive mismatches:', mismatches)
    for key, s in summary.items():
        print(f'{key:18s}', s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['quick', 'quick_report', 'run', 'evaluate'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.command in ('quick', 'run'):
        globals()[args.command](args.workers)
    else:
        globals()[args.command]()
