"""Post-reparse pruning (P) and the reparse pipeline on RESPACING-9 ciphertext (H), 20,800 letters.

    python -m experiments.length_scaling_v4 run historical prune
    python -m experiments.length_scaling_v4 run historical respacing9
    python -m experiments.length_scaling_v4 report
See experiments/length-scaling-v4/PROTOCOL.md.
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
from experiments.length_scaling_v3 import REPARSE_ROUNDS, REPARSE_WIDTH, OUT as V3
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import solvers
from voynich.context_reparse import reparse
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

OUT = ROOT / 'experiments/length-scaling-v4'
N = 20800
CANDIDATES = ('prune', 'respacing9')


def reparse_rounds(tokens, prior, segmentation, ref, s, seed0):
    rounds = []
    for r in range(REPARSE_ROUNDS):
        rp = reparse(tokens, ref['mapping'], prior, width=REPARSE_WIDTH)
        changed = sum(tuple(a) != tuple(b) for a, b in zip(rp['segmentation'], segmentation))
        segmentation = rp['segmentation']; units = role_units(segmentation)
        ref = refine(units, prior, majority_key(units, rp['recovered']), (), seed=seed0 + r, kicks=s['refine_kicks'],
                     kick_size=s['refine_kick_size'], cap=s['refine_cap'])
        rounds.append(dict(changed_parses=changed, refine_cap_hit=ref['cap_hit']))
    return segmentation, units, ref, rounds


def run(source, candidate):
    target = OUT / f'{source}-{candidate}.json'
    if target.exists(): raise FileExistsError('Already run')
    s = scaled(round_four()['settings'], N, 'sqrt'); prior = CharacterPrior.load(fit_priors()['prose+verse'])
    segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    vendor = load_vendor()
    if candidate == 'respacing9': vendor.RESPACING = 9
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
    pieces = set(rep['candidate_pieces']); extra = {}
    if candidate == 'prune':
        use = Counter(p for parts in segmentation for p in parts)
        kept = {p for p in pieces if use[p] >= s['repair_usage_floor']}
        again = joint_em(tokens, prior, pieces=kept, **em)
        units = role_units(again['segmentation'])
        ref = refine(units, prior, majority_key(units, again['recovered']), (), seed=11, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
        segmentation, units, ref, more = reparse_rounds(tokens, prior, again['segmentation'], ref, s, 12)
        extra = dict(pruned_from=len(pieces), pruned_to=len(kept), second_rounds=more, rerun_cap_hit=again['cap_hit'])
        pieces = kept
    pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'],
                 sweeps=s['polish_sweeps'], cap=s['polish_cap'])
    true_pieces = {p for t in truth for p in t}
    kinds = Counter()
    for parts, tp in zip(segmentation, truth):
        if tuple(parts) != tuple(tp):
            kinds['split->whole' if len(tp) == 2 and len(parts) == 1 else 'whole->split' if len(tp) == 1 else 'split->split'] += 1
    words, length = [], 0
    for w in texts_spaced()[source].split():
        if length + len(w) > N: break
        words.append(w); length += len(w)
    row = dict(letters=N, candidate=candidate, tokens=len(tokens), paired_share=sum(len(t) == 2 for t in truth) / len(truth),
               refined_cer=cer(ref['recovered'], dense), polished_cer=cer(pol['recovered'], dense),
               agreement=sum(tuple(a) == tuple(b) for a, b in zip(segmentation, truth)) / len(truth), misparsed=dict(kinds),
               lexicon=dict(pieces=len(pieces), true=len(pieces & true_pieces), spurious=len(pieces - true_pieces), missing=len(true_pieces - pieces)),
               wer_v3=edit_distance(segmenters['v3'].segment(pol['recovered']).split(), words) / len(words), reparse_rounds=rounds, **extra,
               cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], polish=pol['cap_hit']),
               seconds=time.monotonic() - started)
    print(json.dumps(dict(source=source, candidate=candidate, polished_cer=round(row['polished_cer'], 4), agreement=round(row['agreement'], 4),
                          lexicon=row['lexicon'], seconds=round(row['seconds']))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=source, candidate=candidate, seed=SEED, row=row, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
                                      development_only=True, voynich_used=False, at=time.time()), indent=2) + '\n')


def report():
    sources = ('historical', 'modern', 'verse')
    base = {s: next(r for r in json.loads((V3 / f'{s}-reparse.json').read_text())['rows'] if r['letters'] == N) for s in sources}
    rows = {c: {s: json.loads((OUT / f'{s}-{c}.json').read_text())['row'] for s in sources} for c in CANDIDATES}
    mean = lambda d: sum(d[s]['polished_cer'] for s in sources) / 3
    b, p = mean(base), mean(rows['prune'])
    adopt = b - p >= .003 and all(rows['prune'][s]['polished_cer'] - base[s]['polished_cer'] <= .003 for s in sources)
    out = dict(baseline_R=b, prune=p, respacing9=mean(rows['respacing9']), prune_adopted=adopt,
               table={s: dict(R=base[s]['polished_cer'], prune=rows['prune'][s]['polished_cer'], respacing9=rows['respacing9'][s]['polished_cer'],
                              respacing9_wer=rows['respacing9'][s]['wer_v3'], R_wer=base[s]['wer_v3'], prune_wer=rows['prune'][s]['wer_v3'])
                      for s in sources})
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'report']); p.add_argument('source', nargs='?'); p.add_argument('candidate', nargs='?')
    a = p.parse_args()
    run(a.source, a.candidate) if a.command == 'run' else report()
