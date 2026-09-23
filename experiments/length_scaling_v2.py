"""Length-aware lexicon: round four with count thresholds and caps scaled by ciphertext length.

    python -m experiments.length_scaling_v2 run historical linear   # one process per (source, rule)
    python -m experiments.length_scaling_v2 report
See experiments/length-scaling-v2/PROTOCOL.md.
"""
import argparse
from collections import Counter
import json
import math
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors
from experiments.joint_development_v3 import attribute
from experiments.joint_recovery_v4 import ROOT, verify as round_four
from experiments.length_scaling import texts_spaced, SEED, OUT as V1
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import solvers
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

OUT = ROOT / 'experiments/length-scaling-v2'
LENGTHS = (10400, 20800)
RULES = dict(linear=lambda k: k, sqrt=math.sqrt)


def scaled(s, n, rule):
    k = n / 5200; f = RULES[rule](k)
    return dict(s, minimum=round(6 * f), prune_usage=3 * f, repair_minimum=max(2, round(2 * f)), repair_usage_floor=1 * f,
                refine_cap=300 * k, polish_cap=1200 * k)


def run(source, rule):
    target = OUT / f'{source}-{rule}.json'
    if target.exists(): raise FileExistsError('Already run')
    base = round_four()['settings']; prior = CharacterPrior.load(fit_priors()['prose+verse'])
    segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    vendor = load_vendor(); text = texts()[source]; rows = []
    for n in LENGTHS:
        s = scaled(base, n, rule); dense = text[:n]
        tokens, truth = encode(vendor, dense, SEED); truth = [tuple(t) for t in truth]
        cap = 3600 * n / 5200; started = time.monotonic()
        em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
        first = joint_em(tokens, prior, **em)
        second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
        rep = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                     usage_floor=s['repair_usage_floor'], **em)
        units = role_units(rep['segmentation'])
        ref = refine(units, prior, majority_key(units, rep['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                     cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
        pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'],
                     sweeps=s['polish_sweeps'], cap=min(s['polish_cap'], max(30., cap - (time.monotonic() - started))))
        true_pieces = {p for t in truth for p in t}; pieces = set(rep['candidate_pieces'])
        kinds = Counter()
        for parts, tp in zip(rep['segmentation'], truth):
            if tuple(parts) != tuple(tp):
                kinds['split->whole' if len(tp) == 2 and len(parts) == 1 else 'whole->split' if len(tp) == 1 else 'split->split'] += 1
        words, length = [], 0
        for w in texts_spaced()[source].split():
            if length + len(w) > n: break
            words.append(w); length += len(w)
        rows.append(dict(letters=n, settings={k: s[k] for k in ('minimum', 'prune_usage', 'repair_minimum', 'repair_usage_floor', 'refine_cap', 'polish_cap')},
                         tokens=len(tokens), refined_cer=cer(ref['recovered'], dense), polished_cer=cer(pol['recovered'], dense),
                         agreement=sum(tuple(a) == tuple(b) for a, b in zip(rep['segmentation'], truth)) / len(truth), misparsed=dict(kinds),
                         **attribute(rep['segmentation'], truth, ref['recovered'], dense),
                         lexicon=dict(pieces=len(pieces), true=len(pieces & true_pieces), spurious=len(pieces - true_pieces), missing=len(true_pieces - pieces)),
                         wer_v3=edit_distance(segmenters['v3'].segment(pol['recovered']).split(), words) / len(words),
                         cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], refine=ref['cap_hit'], polish=pol['cap_hit']),
                         seconds=time.monotonic() - started))
        print(json.dumps(dict(source=source, rule=rule, letters=n, polished_cer=round(rows[-1]['polished_cer'], 4),
                              lexicon=rows[-1]['lexicon'], seconds=round(rows[-1]['seconds']))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=source, rule=rule, seed=SEED, rows=rows, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
                                      development_only=True, voynich_used=False, at=time.time()), indent=2) + '\n')


def report():
    sources = ('historical', 'modern', 'verse')
    base = {s: next(r for r in json.loads((V1 / f'{s}.json').read_text())['rows'] if r['letters'] == 5200) for s in sources}
    out = {}
    for rule in RULES:
        rows = {s: {r['letters']: r for r in json.loads((OUT / f'{s}-{rule}.json').read_text())['rows']} for s in sources}
        for s in sources: rows[s][5200] = base[s]
        mean = {n: sum(rows[s][n]['polished_cer'] for s in sources) / 3 for n in (5200, 10400, 20800)}
        out[rule] = dict(mean_polished_cer=mean, long_passage_round=mean[20800] <= mean[5200] / 2,
                         table={s: {n: {k: rows[s][n][k] for k in ('polished_cer', 'agreement', 'wer_v3', 'lexicon', 'cap_hits')} for n in (5200, 10400, 20800)} for s in sources})
    out['decision'] = dict(primary='linear', long_passage_round=out['linear']['long_passage_round'])
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({r: out[r]['mean_polished_cer'] for r in RULES}, indent=2), out['decision'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'report']); p.add_argument('source', nargs='?'); p.add_argument('rule', nargs='?')
    a = p.parse_args()
    run(a.source, a.rule) if a.command == 'run' else report()
