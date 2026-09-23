"""Two non-count lexicon/parse candidates against the square-root length baseline.

    python -m experiments.length_scaling_v3 run historical glue      # self-inclusive concatenation test
    python -m experiments.length_scaling_v3 run historical reparse   # context reparse + key refit, twice
    python -m experiments.length_scaling_v3 report
See experiments/length-scaling-v3/PROTOCOL.md.
"""
import argparse
from collections import Counter
import json
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors
from experiments.joint_development_v3 import attribute
from experiments.joint_recovery_v4 import ROOT, verify as round_four
from experiments.length_scaling import texts_spaced, SEED
from experiments.length_scaling_v2 import scaled, OUT as V2
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import solvers
from voynich.context_reparse import reparse
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich import lexicon_repair, lexicon_repair_v2
from voynich.variable_units import refine

OUT = ROOT / 'experiments/length-scaling-v3'
LENGTHS = (10400, 20800)
CANDIDATES = ('glue', 'reparse')
REPARSE_ROUNDS, REPARSE_WIDTH = 2, 128


def run(source, candidate):
    target = OUT / f'{source}-{candidate}.json'
    if target.exists(): raise FileExistsError('Already run')
    base = round_four()['settings']; prior = CharacterPrior.load(fit_priors()['prose+verse'])
    segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    vendor = load_vendor(); text = texts()[source]; rows = []
    repair = lexicon_repair_v2.repair if candidate == 'glue' else lexicon_repair.repair
    for n in LENGTHS:
        s = scaled(base, n, 'sqrt'); dense = text[:n]
        tokens, truth = encode(vendor, dense, SEED); truth = [tuple(t) for t in truth]
        cap = 3600 * n / 5200; started = time.monotonic()
        em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
        first = joint_em(tokens, prior, **em)
        second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
        rep = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                     usage_floor=s['repair_usage_floor'], **em)
        segmentation = rep['segmentation']
        units = role_units(segmentation)
        ref = refine(units, prior, majority_key(units, rep['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                     cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
        rounds = []
        if candidate == 'reparse':
            for r in range(REPARSE_ROUNDS):
                rp = reparse(tokens, ref['mapping'], prior, width=REPARSE_WIDTH)
                changed = sum(tuple(a) != tuple(b) for a, b in zip(rp['segmentation'], segmentation))
                segmentation = rp['segmentation']; units = role_units(segmentation)
                ref = refine(units, prior, majority_key(units, rp['recovered']), (), seed=1 + r + 1, kicks=s['refine_kicks'],
                             kick_size=s['refine_kick_size'], cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
                rounds.append(dict(changed_parses=changed, reparse_seconds=rp['seconds'], cer=cer(ref['recovered'], dense), refine_cap_hit=ref['cap_hit']))
        pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'],
                     sweeps=s['polish_sweeps'], cap=min(s['polish_cap'], max(30., cap - (time.monotonic() - started))))
        true_pieces = {p for t in truth for p in t}; pieces = set(rep['candidate_pieces'])
        kinds = Counter()
        for parts, tp in zip(segmentation, truth):
            if tuple(parts) != tuple(tp):
                kinds['split->whole' if len(tp) == 2 and len(parts) == 1 else 'whole->split' if len(tp) == 1 else 'split->split'] += 1
        words, length = [], 0
        for w in texts_spaced()[source].split():
            if length + len(w) > n: break
            words.append(w); length += len(w)
        rows.append(dict(letters=n, candidate=candidate, settings={k: s[k] for k in ('minimum', 'prune_usage', 'repair_minimum', 'repair_usage_floor', 'refine_cap', 'polish_cap')},
                         tokens=len(tokens), refined_cer=cer(ref['recovered'], dense), polished_cer=cer(pol['recovered'], dense),
                         agreement=sum(tuple(a) == tuple(b) for a, b in zip(segmentation, truth)) / len(truth), misparsed=dict(kinds),
                         **attribute(segmentation, truth, ref['recovered'], dense), reparse_rounds=rounds, repair_record=rep['repair']['record'],
                         lexicon=dict(pieces=len(pieces), true=len(pieces & true_pieces), spurious=len(pieces - true_pieces), missing=len(true_pieces - pieces)),
                         wer_v3=edit_distance(segmenters['v3'].segment(pol['recovered']).split(), words) / len(words),
                         cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], refine=ref['cap_hit'], polish=pol['cap_hit']),
                         seconds=time.monotonic() - started))
        print(json.dumps(dict(source=source, candidate=candidate, letters=n, polished_cer=round(rows[-1]['polished_cer'], 4),
                              agreement=round(rows[-1]['agreement'], 4), lexicon=rows[-1]['lexicon'], seconds=round(rows[-1]['seconds']))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=source, candidate=candidate, seed=SEED, rows=rows, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
                                      development_only=True, voynich_used=False, at=time.time()), indent=2) + '\n')


def choose(means, baseline):
    """Declared selection: at 20,800 at least 0.5 points below baseline and at 10,400 no more than 0.2 above."""
    eligible = [c for c in CANDIDATES if means[c][20800] <= baseline[20800] - .005 and means[c][10400] <= baseline[10400] + .002]
    return min(eligible, key=lambda c: (means[c][20800], c)) if eligible else None


def report():
    sources = ('historical', 'modern', 'verse')
    base_rows = {s: {r['letters']: r for r in json.loads((V2 / f'{s}-sqrt.json').read_text())['rows']} for s in sources}
    baseline = {n: sum(base_rows[s][n]['polished_cer'] for s in sources) / 3 for n in LENGTHS}
    rows = {c: {s: {r['letters']: r for r in json.loads((OUT / f'{s}-{c}.json').read_text())['rows']} for s in sources} for c in CANDIDATES}
    means = {c: {n: sum(rows[c][s][n]['polished_cer'] for s in sources) / 3 for n in LENGTHS} for c in CANDIDATES}
    selected = choose(means, baseline)
    out = dict(baseline_sqrt=baseline, means=means, selected=selected,
               long_passage_round=bool(selected) and means[selected][20800] <= .0281,
               table={c: {s: {n: {k: rows[c][s][n][k] for k in ('polished_cer', 'agreement', 'wer_v3', 'lexicon', 'misparsed', 'cap_hits')}
                              for n in LENGTHS} for s in sources} for c in CANDIDATES})
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(baseline=baseline, means=means, selected=selected, long_passage_round=out['long_passage_round']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'report']); p.add_argument('source', nargs='?'); p.add_argument('candidate', nargs='?')
    a = p.parse_args()
    run(a.source, a.candidate) if a.command == 'run' else report()
