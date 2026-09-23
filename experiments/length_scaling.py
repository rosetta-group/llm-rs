"""Length scaling of round-four Naibbe recovery on development texts (PROTOCOL in experiments/length-scaling/).

    python -m experiments.length_scaling run historical   # one process per source
    python -m experiments.length_scaling report
"""
import argparse
from collections import Counter
import json
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors
from experiments.joint_development_v3 import attribute
from experiments.joint_recovery_v4 import ROOT, verify as round_four
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import solvers
from voynich.data import digest
from voynich.decipher import edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

OUT = ROOT / 'experiments/length-scaling'
LENGTHS = (5200, 10400, 20800)
SEED = 7000


def run(source):
    target = OUT / f'{source}.json'
    if target.exists(): raise FileExistsError('Already run: ' + source)
    s = round_four()['settings']; prior = CharacterPrior.load(fit_priors()['prose+verse'])
    segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    vendor = load_vendor(); text = texts()[source]; rows = []
    reference = None
    for n in LENGTHS:
        dense = text[:n]
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
        # References for WER: whole words inside the prefix (segmentation audit convention).
        words, length = [], 0
        for w in texts_spaced()[source].split():
            if length + len(w) > n: break
            words.append(w); length += len(w)
        segmented = segmenters['v3'].segment(pol['recovered'])
        rows.append(dict(letters=n, tokens=len(tokens), refined_cer=cer(ref['recovered'], dense), polished_cer=cer(pol['recovered'], dense),
                         agreement=sum(tuple(a) == tuple(b) for a, b in zip(rep['segmentation'], truth)) / len(truth), misparsed=dict(kinds),
                         **attribute(rep['segmentation'], truth, ref['recovered'], dense),
                         lexicon=dict(pieces=len(pieces), true=len(pieces & true_pieces), spurious=len(pieces - true_pieces), missing=len(true_pieces - pieces)),
                         wer_v3=edit_distance(segmented.split(), words) / len(words),
                         cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], refine=ref['cap_hit'], polish=pol['cap_hit']),
                         seconds=time.monotonic() - started, cap_seconds=cap))
        print(json.dumps(dict(source=source, letters=n, polished_cer=round(rows[-1]['polished_cer'], 4), agreement=round(rows[-1]['agreement'], 4),
                              seconds=round(rows[-1]['seconds']))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(source=source, seed=SEED, settings=s, rows=rows, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
                                      development_only=True, voynich_used=False, at=time.time()), indent=2) + '\n')


def texts_spaced():
    """Development texts with spaces, in the same order as texts()."""
    from experiments.segmentation_audit import streams
    return streams()


def report():
    rows = {s: json.loads((OUT / f'{s}.json').read_text())['rows'] for s in ('historical', 'modern', 'verse')}
    mean = {n: sum(next(r for r in rs if r['letters'] == n)['polished_cer'] for rs in rows.values()) / 3 for n in LENGTHS}
    decision = mean[20800] <= mean[5200] / 2
    out = dict(mean_polished_cer=mean, long_passage_round=decision,
               table={s: [{k: r[k] for k in ('letters', 'polished_cer', 'refined_cer', 'agreement', 'wer_v3', 'seconds')} for r in rs] for s, rs in rows.items()})
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'report']); p.add_argument('source', nargs='?')
    a = p.parse_args()
    run(a.source) if a.command == 'run' else report()
