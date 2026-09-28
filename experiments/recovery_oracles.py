"""Development diagnostic: where the decoding cost of decoder A comes from, on released positives.

python -m experiments.recovery_oracles run | report

Uses the answers of the released second confirmation, so nothing here is evidence for a method.
For each positive and its true-language prior, the frozen stages and A are rerun (they reproduce the
archived key), then one true component at a time replaces the decoder's own:

  whole     found lexicon + missing true whole-token pieces
  halves    found lexicon + missing true first/second pieces
  lexicon   found lexicon + every missing true piece
  segment   true segmentation, letters refined from A's key
  letters   A's final segmentation, true letter for every unit the truth defines
  truth     true segmentation and true letters

Each variant ends like A (refine, then two reparse rounds, except `truth` and `letters`, which are
scored as given) and is scored on both passages: fit excess, transfer excess and CER.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import time

from experiments import key_recovery_confirmation_v2 as confirmation
from experiments import key_recovery_development as development
from experiments import rejection_transfer_v2 as previous
from experiments.joint_development import load_vendor, majority_key, role_units
from experiments.rejection_transfer_v2 import read, seal
from voynich.context_reparse import reparse
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.rejection import transfer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/recovery-oracles'
STATE = ROOT / 'artifacts/recovery-oracles'
VARIANTS = ('A', 'whole', 'halves', 'lexicon', 'segment', 'letters', 'truth')


def traced(vendor, plaintext, key, seed):
    """Tokens plus the true segmentation and letters, from the encoder trace."""
    random.seed(seed)
    trace = io.StringIO()
    tokens = vendor.encrypt_naibbe(''.join(key[c] for c in plaintext), vendor.naibbe_tables,
                                   vendor.placeholder_to_glyph, use_78=False, pre_plaintext_file=trace)
    glyphs = set(vendor.placeholder_to_glyph.values())
    inverse = {v: k for k, v in key.items()}
    segmentation, letters = [], []
    for token, chunk in zip(tokens, trace.getvalue().split()):
        parts = (token,) if len(chunk) == 1 else \
            [(token[:i], token[i:]) for i in range(1, len(token)) if token[:i] in glyphs and token[i:] in glyphs][0]
        segmentation.append(tuple(parts))
        letters.extend(inverse[c] for c in chunk)
    if ''.join(letters) != plaintext:
        raise ValueError('Trace round trip failed')
    return tokens, segmentation, ''.join(letters)


def unit_key(segmentation, letters):
    votes = defaultdict(Counter)
    for unit, letter in zip(role_units(segmentation), letters):
        votes[unit][letter] += 1
    return {u: c.most_common(1)[0][0] for u, c in votes.items()}


def finish(tokens, segmentation, key, prior, f, seed):
    """Refine, then two reparse rounds, as the end of A."""
    units = role_units(segmentation)
    ref = development.refit(units, prior, {u: key.get(u, 'a') for u in set(units)}, seed, f, f['refine_cap'])
    for r in range(development.REPARSE_ROUNDS):
        parsed = reparse(tokens, ref['mapping'], prior, width=development.REPARSE_WIDTH)
        units = role_units(parsed['segmentation'])
        ref = development.refit(units, prior, majority_key(units, parsed['recovered']), seed + 1 + r, f, f['refine_cap'])
    return ref['mapping'], ref['recovered']


def with_pieces(tokens, prior, pieces, f):
    s = f['settings']
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=900)
    result = previous.joint_em(tokens, prior, pieces=set(pieces), **em)
    segmentation = [tuple(p) for p in result['segmentation']]
    return segmentation, majority_key(role_units(segmentation), result['recovered'])


def case(ident):
    path = STATE / f'{ident}.json'
    if path.exists():
        return str(path)
    freeze = read(confirmation.OUT / 'freeze.json')
    answer = read(confirmation.STATE / 'evaluator-only/answers.json')[ident]
    model = freeze['labels'][answer['language']]
    prior = CharacterPrior.load(confirmation.prior_path(model))
    entropy = freeze['entropy'][model]
    f = read(confirmation.PRIOR_FREEZE)
    vendor = load_vendor()
    fit_tokens, fit_seg, fit_text = traced(vendor, answer['passages'][0]['plaintext'], answer['key'], answer['seeds'][0])
    held_tokens, _, held_text = traced(vendor, answer['passages'][1]['plaintext'], answer['key'], answer['seeds'][1])
    public = read(confirmation.STATE / 'public' / f'{ident}-fit.json')['ciphertext'].split()
    if public != fit_tokens:
        raise ValueError('Regenerated ciphertext differs from the sealed input')
    s, started = f['settings'], time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=f['fit_cap'] * .25)
    first = previous.joint_em(fit_tokens, prior, **em)
    second = previous.prune_and_rerun(fit_tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = previous.repair(fit_tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'],
                               passes=s['repair_passes'], usage_floor=s['repair_usage_floor'], **em)
    found = set(repaired['candidate_pieces'])
    truth_key = unit_key(fit_seg, fit_text)
    true_whole = {parts[0] for parts in fit_seg if len(parts) == 1}
    true_halves = {p for parts in fit_seg if len(parts) == 2 for p in parts}
    archived = read(confirmation.STATE / 'fit' / f'{ident}-{model}.json')['result']['A']
    keys = {}
    # A itself, rebuilt so its final segmentation is available.
    A = development.fit_both(fit_tokens, prior, f)['A']
    if A['mapping'] != archived['mapping']:
        raise ValueError('A did not reproduce the archived key')
    keys['A'] = (A['mapping'], A['recovered'])
    a_seg = [tuple(p) for p in reparse(fit_tokens, A['mapping'], prior, width=development.REPARSE_WIDTH)['segmentation']]
    for name, extra in (('whole', true_whole - found), ('halves', true_halves - found),
                        ('lexicon', (true_whole | true_halves) - found)):
        segmentation, key = with_pieces(fit_tokens, prior, found | extra, f)
        keys[name] = finish(fit_tokens, segmentation, key, prior, f, 40)
    keys['segment'] = finish(fit_tokens, fit_seg, A['mapping'], prior, f, 60)
    letters_key = {u: truth_key.get(u, A['mapping'].get(u, 'a')) for u in set(role_units(a_seg))}
    keys['letters'] = (letters_key, ''.join(letters_key[u] for u in role_units(a_seg)))
    keys['truth'] = (truth_key, fit_text)
    true_transfer = prior.bits(held_text) / len(held_text) - entropy
    true_fit = prior.bits(fit_text) / len(fit_text) - entropy
    rows = {}
    for name, (mapping, recovered) in keys.items():
        held = transfer(held_tokens, mapping, prior)
        rows[name] = dict(fit_excess=prior.bits(recovered) / len(recovered) - entropy,
                          transfer_excess=held['bits_per_letter'] - entropy,
                          fit_cer=edit_distance(recovered, fit_text) / len(fit_text),
                          transfer_cer=edit_distance(held['recovered'], held_text) / len(held_text),
                          coverage=held['token_coverage'])
        rows[name]['transfer_cost'] = rows[name]['transfer_excess'] - true_transfer
        rows[name]['fit_cost'] = rows[name]['fit_excess'] - true_fit
    seal(path, dict(id=ident, block=answer['block'], language=answer['language'], model=model,
                    missing=dict(whole=len(true_whole - found), halves=len(true_halves - found)),
                    true_fit_excess=true_fit, true_transfer_excess=true_transfer, variants=rows,
                    seconds=time.monotonic() - started, development_only=True))
    print(f"{answer['block']} {answer['language']} done", flush=True)
    return str(path)


def run(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    answers = read(confirmation.STATE / 'evaluator-only/answers.json')
    ids = sorted((i for i, a in answers.items() if a['kind'] == 'positive'), key=lambda i: answers[i]['block'])
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(case, ids))


def report():
    rows = [read(p) for p in sorted(STATE.glob('*.json'))]
    summary = {}
    for name in VARIANTS:
        summary[name] = {k: statistics.median(r['variants'][name][k] for r in rows)
                         for k in ('transfer_cost', 'fit_cost', 'transfer_cer', 'fit_cer')}
    seal(OUT / 'results.json', dict(rows=rows, median=summary, cases=len(rows), development_only=True, at=time.time()))
    for name, s in summary.items():
        print(f"{name:8s} transfer cost {s['transfer_cost']:+.3f}  fit cost {s['fit_cost']:+.3f}  "
              f"transfer CER {100 * s['transfer_cer']:.1f}%  fit CER {100 * s['fit_cer']:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'report'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    run(args.workers) if args.command == 'run' else report()
