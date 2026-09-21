"""Development record: codebook-free recovery of Naibbe ciphertext by joint segmentation and EM.

Development data only. Historical text is the held-out development tales (Novellino and
Decameron, every fifth tale); modern text is UD Italian-ISDT dev. The frozen prior from
`artifacts/standard-decipherment/prior.npz` is reused unchanged. Nothing here touches
evaluation passages, Voynich pages, or the sealed final test. The published Naibbe encoder
is used only to make development ciphertext; its trace is opened only for the oracle
diagnostics, which are labelled as such.

    .venv/bin/python -m experiments.joint_development            # full table, about an hour of CPU
    .venv/bin/python -m experiments.joint_development --quick    # two conditions, for a smoke check
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import time

from experiments.historical_sources import verify as historical
from experiments.segmentation import corpus
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.homophonic import hmm_em
from voynich.joint_segments import joint_em, candidate_pieces
from voynich.variable_units import induce_pieces, refine, objective, variable_beam

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/joint-development'
PRIOR = ROOT / 'artifacts/standard-decipherment/prior.npz'
CODE = ['voynich/joint_segments.py', 'voynich/variable_units.py', 'voynich/homophonic.py',
        'voynich/description_length.py', 'experiments/joint_development.py']
SETTINGS = dict(minimum=6, joint_restarts=4, joint_iterations=60, em_restarts=6, em_iterations=200,
                refine_kicks=30, refine_kick_size=6, refine_cap=300, warm_rounds=1, warm_iterations=30, seed=3)


def load_vendor():
    vendor_dir = ROOT / 'artifacts/decipherment/vendor'
    for f in json.loads((ROOT / 'experiments/decipherment-sources.json').read_text())['files']:
        if digest(ROOT / f['path']) != f['sha256']:
            raise ValueError('Encoder drift')
    spec = importlib.util.spec_from_file_location('naibbe_published', vendor_dir / 'naibbe.py')
    vendor = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(vendor_dir); spec.loader.exec_module(vendor)
    finally:
        os.chdir(cwd)
    return vendor


def development_text(source):
    if source == 'historical':
        return normalize(' '.join(' '.join(r['paragraphs']) for r in historical() if r['split'] == 'dev')).replace(' ', '')
    rows, _ = corpus('UD_Italian-ISDT', 'dev')
    return normalize(' '.join(' '.join(r['words']) for r in rows)).replace(' ', '')


def encode(vendor, dense, seed):
    """Naibbe ciphertext with a hidden letter permutation; the trace is oracle-only."""
    rng = random.Random(seed); alphabet = list(ALPHABET); rng.shuffle(alphabet)
    key = dict(zip(ALPHABET, alphabet)); coded = ''.join(key[c] for c in dense)
    random.seed(seed); trace = io.StringIO()
    tokens = vendor.encrypt_naibbe(coded, vendor.naibbe_tables, vendor.placeholder_to_glyph, use_78=False, pre_plaintext_file=trace)
    glyphs = set(vendor.placeholder_to_glyph.values())
    truth = []
    for tok, chunk in zip(tokens, trace.getvalue().split()):
        if len(chunk) == 1:
            truth.append((tok,))
        else:
            truth.append([(tok[:i], tok[i:]) for i in range(1, len(tok)) if tok[:i] in glyphs and tok[i:] in glyphs][0])
    return tokens, truth


def role_units(segmentation):
    return [f'{role}:{piece}' for parts in segmentation
            for piece, role in zip(parts, ('u',) if len(parts) == 1 else ('p', 's'))]


def majority_key(units, letters):
    votes = defaultdict(Counter)
    for u, c in zip(units, letters):
        if c != '?':
            votes[u][c] += 1
    return {u: (votes[u].most_common(1)[0][0] if votes[u] else 'a') for u in set(units)}


def cer(text, gold):
    return edit_distance(text, gold) / len(gold)


def oracle_stage(units, truth_mapping, dense, prior, s):
    """Diagnostic only: mapping search with the true piece segmentation supplied."""
    started = time.time()
    tb = objective(units, prior, truth_mapping)
    beam = variable_beam(units, prior, (), width=512, cap=900)
    em = hmm_em(units, prior, restarts=s['em_restarts'], iterations=s['em_iterations'], seed=7, cap=1500)
    key = majority_key(units, em['recovered'])
    ref = refine(units, prior, key, (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
    return dict(unit_types=len(set(units)), truth_bits=tb, beam_bits=beam['search_bits'], beam_cer=cer(beam['recovered'], dense),
                em_cer=cer(em['recovered'], dense), em_key_bits=objective(units, prior, key),
                em_refined_cer=cer(ref['recovered'], dense), em_refined_bits=ref['search_bits'], seconds=time.time() - started)


def codebook_free_stage(tokens, truth, dense, prior, s):
    """The actual method: no trace, no codebook, no key."""
    started = time.time()
    agreement = lambda seg: sum(tuple(a) == tuple(b) for a, b in zip(seg, truth)) / len(tokens)
    induced = induce_pieces(tokens, roles=True)
    result = dict(deterministic_segmentation_agreement=agreement([induced['segmentation'][t] for t in tokens]))
    joint = joint_em(tokens, prior, minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=3000)
    pieces = candidate_pieces(tokens, s['minimum'])
    seg, letters = joint['segmentation'], joint['recovered']
    result.update(joint_pieces=joint['pieces'], joint_parses=joint['parses'], joint_restart_scores=joint['restart_scores'],
                  joint_segmentation_agreement=agreement(seg), joint_cer=cer(letters, dense), joint_recovered_letters=len(letters))
    rounds = []
    for r in range(s['warm_rounds'] + 1):
        units = role_units(seg)
        ref = refine(units, prior, majority_key(units, letters), (), seed=1 + r, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'], cap=s['refine_cap'])
        row = dict(refined_cer=cer(ref['recovered'], dense), refined_bits=ref['search_bits'], segmentation_agreement=agreement(seg))
        final = ref['recovered']
        if r < s['warm_rounds']:
            # re-run the joint EM around the refined key; posterior parses replace the segmentation
            warm = joint_em(tokens, prior, minimum=s['minimum'], restarts=1, iterations=s['warm_iterations'], seed=s['seed'] + 1 + r,
                            cap=1800, warm_start=ref['mapping'])
            seg, letters = warm['segmentation'], warm['recovered']
            row.update(warm_cer=cer(letters, dense), warm_segmentation_agreement=agreement(seg), warm_log_likelihood=warm['log_likelihood'])
        rounds.append(row)
    result.update(rounds=rounds, final_cer=cer(final, dense), final_recovered=final[:120], seconds=time.time() - started)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Development results exist; remove the folder to regenerate')
    prior = CharacterPrior.load(PRIOR)
    vendor = load_vendor()
    conditions = [('historical', 5200), ('modern', 5200)] if args.quick else \
        [(source, n) for source in ('historical', 'modern') for n in (1300, 2600, 5200, 10400)]
    rows = []
    for source, n in conditions:
        text = development_text(source)
        if len(text) < n:
            raise ValueError('Not enough development text')
        dense = text[:n]
        tokens, truth = encode(vendor, dense, 1000 + n)
        units = role_units(truth); pos = 0; truth_mapping = {}
        for u in units:
            truth_mapping.setdefault(u, dense[pos]); pos += 1
        row = dict(source=source, letters=n, tokens=len(tokens), token_types=len(set(tokens)),
                   true_pieces=len({p for t in truth for p in t}),
                   oracle_segmentation=oracle_stage(units, truth_mapping, dense, prior, SETTINGS),
                   codebook_free=codebook_free_stage(tokens, truth, dense, prior, SETTINGS))
        rows.append(row)
        print(json.dumps(dict(source=source, letters=n, oracle_em_refined_cer=round(row['oracle_segmentation']['em_refined_cer'], 3),
                              codebook_free_final_cer=round(row['codebook_free']['final_cer'], 3),
                              joint_segmentation_agreement=round(row['codebook_free']['joint_segmentation_agreement'], 3))), flush=True)
    output = dict(settings=SETTINGS, prior_sha256=digest(PRIOR), code_sha256={p: digest(ROOT / p) for p in CODE},
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  development_data=dict(historical='Novellino/Decameron development tales', modern='UD_Italian-ISDT dev'),
                  evaluation_passages_used=False, voynich_text_used=False, final_test_scored=False, quick=args.quick,
                  rows=rows, at=time.time())
    (OUT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Wrote', OUT / 'results.json')


if __name__ == '__main__':
    main()
