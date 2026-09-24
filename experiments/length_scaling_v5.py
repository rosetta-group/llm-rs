"""Context admission of rare pieces after the reparse, at RESPACING 17 and 9, 20,800 letters.

    python -m experiments.length_scaling_v5 run historical 17
    python -m experiments.length_scaling_v5 report
See experiments/length-scaling-v5/PROTOCOL.md.
"""
import argparse
from collections import Counter
import json
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors
from experiments.joint_recovery_v4 import ROOT, verify as round_four
from experiments.length_scaling import texts_spaced, SEED
from experiments.length_scaling_v2 import scaled
from experiments.length_scaling_v3 import OUT as V3
from experiments.length_scaling_v4 import OUT as V4, reparse_rounds
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import solvers
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.piece_admission import admit
from voynich.variable_units import refine

OUT = ROOT / 'experiments/length-scaling-v5'
N = 20800
THRESHOLD, MAX_COUNT = 10., 5


def run(source, respacing):
    target = OUT / f'{source}-r{respacing}.json'
    if target.exists(): raise FileExistsError('Already run')
    s = scaled(round_four()['settings'], N, 'sqrt'); prior = CharacterPrior.load(fit_priors()['prose+verse'])
    segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    vendor = load_vendor(); vendor.RESPACING = respacing
    dense = texts()[source][:N]
    tokens, truth = encode(vendor, dense, SEED); truth = [tuple(t) for t in truth]
    cap = 3600 * N / 5200; started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    rep = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                 usage_floor=s['repair_usage_floor'], **em)
    units = role_units(rep['segmentation'])
    ref = refine(units, prior, majority_key(units, rep['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    segmentation, units, ref, rounds = reparse_rounds(tokens, prior, rep['segmentation'], ref, s, 2)
    before_cer = cer(ref['recovered'], dense)
    mapping, record = admit(tokens, segmentation, ref['mapping'], prior, max_count=MAX_COUNT, threshold=THRESHOLD)
    ref = dict(ref, mapping=mapping)
    segmentation, units, ref, more = reparse_rounds(tokens, prior, segmentation, ref, s, 12)
    pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'],
                 sweeps=s['polish_sweeps'], cap=s['polish_cap'])
    true_units = {('u:' if len(t) == 1 else r) + p for t in truth for p, r in zip(t, ('u:',) if len(t) == 1 else ('p:', 's:'))}
    admitted_true = sum(u in true_units for u in record['admitted_units'])
    used = {u for parts in segmentation for u in (['u:' + parts[0]] if len(parts) == 1 else ['p:' + parts[0], 's:' + parts[1]])}
    kinds = Counter()
    for parts, tp in zip(segmentation, truth):
        if tuple(parts) != tuple(tp):
            kinds['split->whole' if len(tp) == 2 and len(parts) == 1 else 'whole->split' if len(tp) == 1 else 'split->split'] += 1
    words, length = [], 0
    for w in texts_spaced()[source].split():
        if length + len(w) > N: break
        words.append(w); length += len(w)
    row = dict(letters=N, respacing=respacing, tokens=len(tokens), cer_before_admission=before_cer,
               refined_cer=cer(ref['recovered'], dense), polished_cer=cer(pol['recovered'], dense),
               agreement=sum(tuple(a) == tuple(b) for a, b in zip(segmentation, truth)) / len(truth), misparsed=dict(kinds),
               admission=dict(record, admitted_true_units=admitted_true), missing_true_units=len(true_units - used),
               wer_v3=edit_distance(segmenters['v3'].segment(pol['recovered']).split(), words) / len(words),
               reparse_rounds=rounds + more, cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], polish=pol['cap_hit']),
               seconds=time.monotonic() - started)
    print(json.dumps(dict(source=source, respacing=respacing, polished_cer=round(row['polished_cer'], 4), before=round(before_cer, 4),
                          admitted=record['admitted'], admitted_true=admitted_true, seconds=round(row['seconds']))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=source, respacing=respacing, seed=SEED, row=row, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
                                      development_only=True, voynich_used=False, at=time.time()), indent=2) + '\n')


def baselines():
    sources = ('historical', 'modern', 'verse')
    r17 = {s: next(r for r in json.loads((V3 / f'{s}-reparse.json').read_text())['rows'] if r['letters'] == N)['polished_cer'] for s in sources}
    r9 = {s: json.loads((V4 / f'{s}-respacing9.json').read_text())['row']['polished_cer'] for s in sources}
    return {17: r17, 9: r9}


def report():
    sources = ('historical', 'modern', 'verse'); base = baselines()
    rows = {r: {s: json.loads((OUT / f'{s}-r{r}.json').read_text())['row'] for s in sources} for r in (17, 9)}
    mean = {r: dict(baseline=sum(base[r].values()) / 3, admission=sum(rows[r][s]['polished_cer'] for s in sources) / 3) for r in (17, 9)}
    better = any(m['baseline'] - m['admission'] >= .003 for m in mean.values())
    safe = all(rows[r][s]['polished_cer'] - base[r][s] <= .002 for r in (17, 9) for s in sources)
    out = dict(mean=mean, adopted=better and safe,
               table={r: {s: dict(baseline=base[r][s], admission=rows[r][s]['polished_cer'], wer=rows[r][s]['wer_v3'],
                                  admitted=rows[r][s]['admission']['admitted'], admitted_true=rows[r][s]['admission']['admitted_true_units'],
                                  missing_true_units=rows[r][s]['missing_true_units']) for s in sources} for r in (17, 9)})
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'report']); p.add_argument('source', nargs='?'); p.add_argument('respacing', nargs='?', type=int)
    a = p.parse_args()
    run(a.source, a.respacing) if a.command == 'run' else report()
