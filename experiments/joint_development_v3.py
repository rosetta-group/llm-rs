"""Round-three development record: a word-level polish after the round-two pipeline.

Development data only (same three 5,200-letter texts as round two). Reruns the frozen
round-two pipeline with the prose+verse prior, then applies the lexical polish at several
weights and attributes the remaining error to mis-parsed tokens versus wrong letters in
correctly parsed tokens. The attribution uses the encoder trace and is evaluator-side only.

    .venv/bin/python -m experiments.joint_development_v3
"""
from pathlib import Path
import json
import subprocess
import time

from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer
from experiments.joint_development_v2 import texts, fit_priors, SETTINGS as ROUND_TWO
from voynich.data import digest
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.segmentation import Segmenter
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/joint-development-v3'
STD = ROOT / 'artifacts/standard-decipherment'
CODE = ['voynich/joint_segments.py', 'voynich/joint_segments_v2.py', 'voynich/variable_units.py', 'voynich/lexical_polish.py',
        'voynich/segmentation.py', 'voynich/description_length.py', 'experiments/joint_development.py',
        'experiments/joint_development_v2.py', 'experiments/joint_development_v3.py']
SETTINGS = dict(ROUND_TWO, polish_weight=1., polish_radius=30, polish_shortlist=5, polish_sweeps=3, polish_cap=1200)
WEIGHTS = (.5, 1., 2., 4.)


def segmenter():
    dev = json.loads((ROOT / 'experiments/standard-decipherment/development.json').read_text())
    return Segmenter(json.loads((STD / 'segmenter.json').read_text()), **dev['selected_segmenter']['parameters'])


def round_two(tokens, prior, s):
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=3000)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    units = role_units(second['segmentation'])
    ref = refine(units, prior, majority_key(units, second['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
    return dict(segmentation=second['segmentation'], units=units, mapping=ref['mapping'], recovered=ref['recovered'])


def attribute(segmentation, truth, recovered, dense):
    """Split letter errors between correctly parsed tokens (aligned letter by letter) and mis-parsed tokens."""
    pos_r = pos_g = 0; ok_tokens = ok_letters = wrong_in_ok = letters_in_bad = 0
    for parts, tp in zip(segmentation, truth):
        lr, lg = len(parts), len(tp)
        if tuple(parts) == tuple(tp):
            ok_tokens += 1; ok_letters += lg
            wrong_in_ok += sum(a != b for a, b in zip(recovered[pos_r:pos_r + lr], dense[pos_g:pos_g + lg]))
        else:
            letters_in_bad += lg
        pos_r += lr; pos_g += lg
    return dict(tokens_correct=ok_tokens / len(truth), letters_in_correct_tokens=ok_letters, wrong_letters_in_correct_tokens=wrong_in_ok,
                error_rate_in_correct_tokens=wrong_in_ok / max(1, ok_letters), letters_in_misparsed_tokens=letters_in_bad,
                share_of_letters_misparsed=letters_in_bad / len(dense))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Round-three development results exist; remove the folder to regenerate')
    prior = CharacterPrior.load(fit_priors()['prose+verse'])
    lexical = LexicalCost(segmenter())
    vendor = load_vendor()
    rows = []
    for source, text in texts().items():
        n = 5200; dense = text[:n]
        tokens, truth = encode(vendor, dense, 1000 + n)
        started = time.time()
        base = round_two(tokens, prior, SETTINGS)
        row = dict(source=source, letters=n, round_two=dict(cer=cer(base['recovered'], dense), **attribute(base['segmentation'], truth, base['recovered'], dense)),
                   polish={}, round_two_seconds=time.time() - started)
        for weight in WEIGHTS:
            started = time.time()
            pol = polish(base['units'], prior, base['mapping'], lexical, weight=weight, radius=SETTINGS['polish_radius'],
                         shortlist=SETTINGS['polish_shortlist'], sweeps=SETTINGS['polish_sweeps'], cap=SETTINGS['polish_cap'])
            row['polish'][str(weight)] = dict(cer=cer(pol['recovered'], dense), changes=len(pol['changes']), sweeps=pol['sweeps'],
                                              lexical_bits=lexical.cost(pol['recovered']), seconds=time.time() - started, cap_hit=pol['cap_hit'],
                                              **attribute(base['segmentation'], truth, pol['recovered'], dense), sample=pol['recovered'][:120])
        row['true_text_lexical_bits'] = lexical.cost(dense)
        rows.append(row)
        print(json.dumps(dict(source=source, round_two_cer=round(row['round_two']['cer'], 3),
                              polished={w: round(v['cer'], 3) for w, v in row['polish'].items()},
                              error_in_correct_tokens_at_1=round(row['polish']['1.0']['error_rate_in_correct_tokens'], 3))), flush=True)
    output = dict(settings=SETTINGS, weights=WEIGHTS, prior_sha256=digest(fit_priors()['prose+verse']), segmenter_sha256=digest(STD / 'segmenter.json'),
                  code_sha256={p: digest(ROOT / p) for p in CODE},
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  evaluation_passages_used=False, voynich_text_used=False, final_test_scored=False, rows=rows, at=time.time())
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Wrote', OUT / 'results.json')


if __name__ == '__main__':
    main()
