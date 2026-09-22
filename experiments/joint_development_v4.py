"""Round-four development record: lexicon repair after the round-two pipeline.

Development data only (the same three 5,200-letter texts as rounds two and three). Runs the
frozen round-two pipeline with the prose+verse prior, then repairs the candidate lexicon
(`voynich/lexicon_repair.py`) under a small grid, refines the key, and records parse agreement,
character error and the lexicon's true/spurious/missing counts against the encoder trace,
which is oracle-only. The setting with the lowest mean character error over the three texts is
selected and polished with the round-three lexical polish. Oracle runs with the true lexicon are
recorded as the ceiling.

    .venv/bin/python -m experiments.joint_development_v4
"""
from collections import Counter
from pathlib import Path
import json
import subprocess
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors, SETTINGS as ROUND_TWO
from experiments.joint_development_v3 import attribute, segmenter
from voynich.data import digest
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/joint-development-v4'
STD = ROOT / 'artifacts/standard-decipherment'
CODE = ['voynich/joint_segments.py', 'voynich/joint_segments_v2.py', 'voynich/lexicon_repair.py', 'voynich/variable_units.py',
        'voynich/lexical_polish.py', 'voynich/segmentation.py', 'voynich/description_length.py', 'experiments/joint_development.py',
        'experiments/joint_development_v2.py', 'experiments/joint_development_v3.py', 'experiments/joint_development_v4.py']
SETTINGS = dict(ROUND_TWO, polish_weight=1., polish_radius=30, polish_shortlist=5, polish_sweeps=3, polish_cap=1200,
                repair_theta=5., repair_usage_floor=1.)
GRID = [dict(minimum=m, passes=k) for m in (1, 2) for k in (1, 2)]


def em_settings(s, cap=3000):
    return dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap)


def refined(tokens, prior, result, s):
    units = role_units(result['segmentation'])
    ref = refine(units, prior, majority_key(units, result['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
    return units, ref


def lexicon_report(pieces, truth):
    true_pieces = {p for t in truth for p in t}
    return dict(pieces=len(pieces), true=len(pieces & true_pieces), spurious=len(pieces - true_pieces), missing=len(true_pieces - pieces))


def parse_report(segmentation, truth):
    kinds = Counter()
    for parts, tp in zip(segmentation, truth):
        if tuple(parts) != tuple(tp):
            kinds['split->whole' if len(tp) == 2 and len(parts) == 1 else 'whole->split' if len(tp) == 1 else 'split->split'] += 1
    return dict(agreement=sum(tuple(a) == tuple(b) for a, b in zip(segmentation, truth)) / len(truth), misparsed=dict(kinds))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Round-four development results exist; remove the folder to regenerate')
    prior = CharacterPrior.load(fit_priors()['prose+verse'])
    lexical = LexicalCost(segmenter())
    vendor = load_vendor()
    s = SETTINGS; em = em_settings(s)
    rows = []
    for source, text in texts().items():
        n = 5200; dense = text[:n]
        tokens, truth = encode(vendor, dense, 1000 + n)
        truth = [tuple(t) for t in truth]
        started = time.time()
        first = joint_em(tokens, prior, **em)
        second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
        units, ref = refined(tokens, prior, second, s)
        row = dict(source=source, letters=n, tokens=len(tokens),
                   round_two=dict(cer=cer(ref['recovered'], dense), **parse_report(second['segmentation'], truth),
                                  lexicon=lexicon_report(set(second['candidate_pieces']), truth), seconds=time.time() - started),
                   repairs={}, oracle={})
        for config in GRID:
            started = time.time()
            rep = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=config['minimum'], passes=config['passes'],
                         usage_floor=s['repair_usage_floor'], **em)
            units, ref = refined(tokens, prior, rep, s)
            key = f"minimum{config['minimum']}_passes{config['passes']}"
            row['repairs'][key] = dict(config, cer=cer(ref['recovered'], dense), **parse_report(rep['segmentation'], truth),
                                       lexicon=lexicon_report(set(rep['candidate_pieces']), truth), record=rep['repair']['record'],
                                       **attribute(rep['segmentation'], truth, ref['recovered'], dense), seconds=time.time() - started,
                                       _units=units, _mapping=ref['mapping'])
            print(json.dumps(dict(source=source, config=key, cer=round(row['repairs'][key]['cer'], 4), agreement=round(row['repairs'][key]['agreement'], 4))), flush=True)
        started = time.time()
        true_pieces = {p for t in truth for p in t}
        oracle = joint_em(tokens, prior, pieces=true_pieces, **em)
        units_o, ref_o = refined(tokens, prior, oracle, s)
        row['oracle'] = dict(true_lexicon=dict(cer=cer(ref_o['recovered'], dense), **parse_report(oracle['segmentation'], truth), seconds=time.time() - started))
        rows.append(row)
    # selection rule, fixed in advance: lowest mean refined CER over the three texts
    keys = list(rows[0]['repairs'])
    means = {k: sum(r['repairs'][k]['cer'] for r in rows) / len(rows) for k in keys}
    selected = min(keys, key=lambda k: (means[k], k))
    for row in rows:
        chosen = row['repairs'][selected]; dense = texts()[row['source']][:row['letters']]
        started = time.time()
        pol = polish(chosen['_units'], prior, chosen['_mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'],
                     shortlist=s['polish_shortlist'], sweeps=s['polish_sweeps'], cap=s['polish_cap'])
        row['selected'] = dict(config=selected, refined_cer=chosen['cer'], polished_cer=cer(pol['recovered'], dense), polish_changes=len(pol['changes']),
                               sample=pol['recovered'][:120], seconds=time.time() - started)
        for r in row['repairs'].values():
            r.pop('_units'); r.pop('_mapping')
        print(json.dumps(dict(source=row['source'], selected=selected, refined_cer=round(chosen['cer'], 4), polished_cer=round(row['selected']['polished_cer'], 4))), flush=True)
    output = dict(settings=SETTINGS, grid=GRID, selected=selected, mean_cer_by_config=means,
                  prior_sha256=digest(fit_priors()['prose+verse']), segmenter_sha256=digest(STD / 'segmenter.json'),
                  code_sha256={p: digest(ROOT / p) for p in CODE},
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  evaluation_passages_used=False, voynich_text_used=False, final_test_scored=False, rows=rows, at=time.time())
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Wrote', OUT / 'results.json')


if __name__ == '__main__':
    main()
