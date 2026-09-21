"""Round-two development record: usage pruning and a verse-augmented prior for codebook-free Naibbe.

Development data only: Novellino/Decameron dev tales, ISDT dev, and Petrarca Canzoniere dev
poems. Round one is experiments/joint_development.py; its module stays frozen. This driver
fits the candidate priors, runs the round-two pipeline (joint EM, usage pruning, joint EM
again, refinement) with each prior on each development text, and records everything.

    .venv/bin/python -m experiments.joint_development_v2
"""
from pathlib import Path
import json
import subprocess
import time

from experiments.historical_sources import verify as historical
from experiments.verse_sources import verify as verse
from experiments.segmentation import corpus
from experiments.joint_development import load_vendor, encode, role_units, majority_key, cer, development_text
from voynich.data import digest
from voynich.decipher import normalize
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/joint-development-v2'
PRIORS = ROOT / 'artifacts/verse-prior'  # ~50 MB of tables; reproducible from the pinned texts, hashed in freezes
CODE = ['voynich/joint_segments.py', 'voynich/joint_segments_v2.py', 'voynich/variable_units.py', 'voynich/homophonic.py',
        'voynich/description_length.py', 'experiments/joint_development.py', 'experiments/joint_development_v2.py',
        'experiments/verse_sources.py']
SETTINGS = dict(minimum=6, joint_restarts=4, joint_iterations=60, prune_usage=3., refine_kicks=30, refine_kick_size=6,
                refine_cap=300, seed=3, verse_weight=3)


def fit_priors():
    """Old prior is the frozen round-one file; the new one adds Petrarca training poems."""
    PRIORS.mkdir(parents=True, exist_ok=True)
    target = PRIORS / 'prior-verse.npz'
    if not target.exists():
        train, _ = corpus('UD_Italian-ISDT', 'train')
        modern = [normalize(' '.join(r['words'])) for r in train]
        prose = [normalize(' '.join(r['paragraphs'])) for r in historical() if r['split'] == 'train']
        poems = [normalize(' '.join(r['lines'])) for r in verse() if r['split'] == 'train']
        CharacterPrior.fit(modern + prose + poems * SETTINGS['verse_weight']).save(target)
    return {'prose': ROOT / 'artifacts/standard-decipherment/prior.npz', 'prose+verse': target}


def texts():
    return {'historical': development_text('historical'), 'modern': development_text('modern'),
            'verse': normalize(' '.join(' '.join(r['lines']) for r in verse() if r['split'] == 'dev')).replace(' ', '')}


def pipeline(tokens, prior, s, cap=3000):
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    stages = {}
    for name, res in (('joint', first), ('pruned', second)):
        units = role_units(res['segmentation'])
        ref = refine(units, prior, majority_key(units, res['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
        stages[name] = dict(segmentation=res['segmentation'], joint_recovered=res['recovered'], refined=ref['recovered'],
                            candidate_pieces=len(res['candidate_pieces']), restart_scores=res['restart_scores'], refined_bits=ref['search_bits'])
    return dict(stages=stages, pruned_from=second['pruned_from'], pruned_to=second['pruned_to'], seconds=time.monotonic() - started)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Round-two development results exist; remove the folder to regenerate')
    priors = {name: CharacterPrior.load(path) for name, path in fit_priors().items()}
    vendor = load_vendor()
    rows = []
    for source, text in texts().items():
        n = 5200
        dense = text[:n]
        tokens, truth = encode(vendor, dense, 1000 + n)
        agreement = lambda seg: sum(tuple(a) == tuple(b) for a, b in zip(seg, truth)) / len(tokens)
        for name, prior in priors.items():
            run = pipeline(tokens, prior, SETTINGS)
            row = dict(source=source, letters=n, prior=name, bits_per_letter_true=prior.bits(dense) / len(dense),
                       pruned_from=run['pruned_from'], pruned_to=run['pruned_to'], seconds=run['seconds'])
            for stage, res in run['stages'].items():
                row[stage] = dict(segmentation_agreement=agreement(res['segmentation']), joint_cer=cer(res['joint_recovered'], dense),
                                  refined_cer=cer(res['refined'], dense), candidate_pieces=res['candidate_pieces'],
                                  restart_scores=res['restart_scores'], refined_bits=res['refined_bits'], sample=res['refined'][:120])
            rows.append(row)
            print(json.dumps(dict(source=source, prior=name, joint_refined_cer=round(row['joint']['refined_cer'], 3),
                                  pruned_refined_cer=round(row['pruned']['refined_cer'], 3),
                                  pruned_agreement=round(row['pruned']['segmentation_agreement'], 3))), flush=True)
    output = dict(settings=SETTINGS, prior_sha256={k: digest(v) for k, v in fit_priors().items()},
                  code_sha256={p: digest(ROOT / p) for p in CODE},
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  development_data=dict(historical='Novellino/Decameron development tales', modern='UD_Italian-ISDT dev',
                                        verse='Petrarca Canzoniere development poems'),
                  evaluation_passages_used=False, voynich_text_used=False, final_test_scored=False, rows=rows, at=time.time())
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Wrote', OUT / 'results.json')


if __name__ == '__main__':
    main()
