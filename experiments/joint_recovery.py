"""Freeze, prepare, solve ciphertext only, then grade: codebook-free Naibbe by joint segmentation and EM.

Follows experiments/standard_decipherment.py. The method and its settings come from
experiments/joint-development/PROTOCOL.md and results.json; nothing is tuned here.

    python -m experiments.joint_recovery freeze     # hashes code, prior, segmenter, protocol, development results
    python -m experiments.joint_recovery verify     # frozen files unchanged and committed
    python -m experiments.joint_recovery prepare    # fresh passages >= 5,200 letters; answers evaluator-only
    python -m experiments.joint_recovery solve      # ciphertext-only predictions, checkpointed per case
    python -m experiments.joint_recovery evaluate   # opens references; writes experiments/joint-recovery/results.json
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import time

from experiments.joint_development import load_vendor, role_units, majority_key
from experiments.segmentation import corpus
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments import joint_em
from voynich.segmentation import Segmenter
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/joint-recovery'
OUT = ROOT / 'experiments/joint-recovery'
DEV = ROOT / 'experiments/joint-development'
STD = ROOT / 'artifacts/standard-decipherment'
CODE = ['voynich/joint_segments.py', 'voynich/variable_units.py', 'voynich/homophonic.py', 'voynich/description_length.py',
        'voynich/segmentation.py', 'voynich/decipher.py', 'experiments/joint_development.py', 'experiments/joint_recovery.py']
INPUTS = [DEV / 'PROTOCOL.md', DEV / 'results.json', STD / 'prior.npz', STD / 'segmenter.json',
          ROOT / 'experiments/standard-decipherment/development.json', ROOT / 'experiments/decipherment-sources.json']
MIN_LETTERS, MAX_LETTERS = 5200, 6000
CASE_CAP = 2400


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def hashes(): return {p: digest(ROOT / p) for p in CODE}


def settings():
    return read(DEV / 'results.json')['settings']


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Already frozen')
    if read(DEV / 'results.json').get('quick'):
        raise ValueError('Development results are the quick smoke table; run the full development first')
    write(OUT / 'freeze.json', dict(code_sha256=hashes(), inputs={str(p.relative_to(ROOT)): digest(p) for p in INPUTS},
                                    settings=settings(), min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, case_cap_seconds=CASE_CAP, at=time.time()))
    print('Frozen; commit before prepare')


def verify(require_commit=True):
    frozen = read(OUT / 'freeze.json')
    if hashes() != frozen['code_sha256']:
        raise ValueError('Frozen code drift')
    for p, h in frozen['inputs'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen input drift: ' + p)
    if require_commit:
        for p in CODE + ['experiments/joint-recovery/freeze.json', 'experiments/joint-development/PROTOCOL.md',
                         'experiments/joint-development/results.json']:
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != digest(ROOT / p):
                raise ValueError('Freeze not committed: ' + p)
    return frozen


def excluded_ids():
    excluded = {'modern': set(), 'historical': set()}
    for r in read(ROOT / 'artifacts/decipherment/evaluator-only/answers.json'):
        excluded['historical'].update(r['source_sentences'])
    for folder in ('segmentation', 'codebook-free', 'standard-decipherment'):
        for r in read(ROOT / f'artifacts/{folder}/evaluator-only/answers.json'):
            excluded[r['dataset']].update(r['source_ids'])
    return excluded


def prepare():
    verify()
    if (STATE / 'public.json').exists():
        raise FileExistsError('Already prepared')
    excluded = excluded_ids()
    seen = set()
    for split in ('train', 'dev'):
        rows, _ = corpus('UD_Italian-ISDT', split)
        seen.update(normalize(' '.join(r['words'])) for r in rows)
    passages = []
    for dataset, repo, split in [('modern', 'UD_Italian-ISDT', 'test'), ('historical', 'UD_Italian-Old', 'train')]:
        rows, source = corpus(repo, split)
        for region in range(2):
            texts, ids, length = [], [], 0
            for row in rows[region * len(rows) // 2:(region + 1) * len(rows) // 2]:
                text = normalize(' '.join(row['words'])); n = len(text.replace(' ', ''))
                if row['id'] in excluded[dataset] or text in seen or not text:
                    continue  # skip excluded sentences; passages are consecutive fresh sentences
                if length + n > MAX_LETTERS:
                    continue
                texts.append(text); ids.append(row['id']); length += n
                if length >= MIN_LETTERS:
                    break
            if length < MIN_LETTERS:
                raise ValueError(f'Not enough fresh {dataset} text in region {region}')
            excluded[dataset].update(ids)
            passages.append(dict(dataset=dataset, source=source, source_ids=ids, plaintext=' '.join(texts),
                                 passage=hashlib.sha256((repo + ':'.join(ids)).encode()).hexdigest()[:16]))
    vendor = load_vendor()
    public, answers = [], []
    for passage in passages:
        dense = passage['plaintext'].replace(' ', '')
        seed = random.SystemRandom().randrange(2 ** 63); rng = random.Random(seed)
        alphabet = list(ALPHABET); rng.shuffle(alphabet); key = dict(zip(ALPHABET, alphabet)); inverse = {v: k for k, v in key.items()}
        coded = ''.join(key[c] for c in dense)
        random.seed(seed); trace = io.StringIO()
        tokens = vendor.encrypt_naibbe(coded, vendor.naibbe_tables, vendor.placeholder_to_glyph, use_78=False, pre_plaintext_file=trace)
        if ''.join(inverse[c] for c in ''.join(trace.getvalue().split())) != dense:
            raise ValueError('Roundtrip failed')
        cipher = ' '.join(tokens)
        ident = hashlib.sha256((cipher + str(seed)).encode()).hexdigest()[:16]
        public.append(dict(id=ident, ciphertext=cipher))
        answers.append(dict(passage, id=ident, family='naibbe', seed=seed, trace=trace.getvalue(), roundtrip=True))
    private = STATE / 'evaluator-only'; private.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
                                         freeze_sha256=digest(OUT / 'freeze.json'),
                                         freeze_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                         cases=len(public), independent_passages=len(passages), letters=[len(p['plaintext'].replace(' ', '')) for p in passages], at=time.time()))
    print(f'Prepared {len(public)} Naibbe cases; references evaluator-only', flush=True)


def decode(tokens, prior, s, cap):
    """The frozen method. Ciphertext tokens in, letters out; nothing else."""
    started = time.monotonic()
    joint = joint_em(tokens, prior, minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .5)
    seg, letters = joint['segmentation'], joint['recovered']
    rounds = []
    for r in range(s['warm_rounds'] + 1):
        units = role_units(seg)
        remaining = max(30., cap - (time.monotonic() - started))
        ref = refine(units, prior, majority_key(units, letters), (), seed=1 + r, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                     cap=min(s['refine_cap'], remaining))
        rounds.append(dict(refined_bits=ref['search_bits'], refine_seconds=ref['seconds'], cap_hit=ref['cap_hit']))
        final = ref['recovered']
        if r < s['warm_rounds']:
            warm = joint_em(tokens, prior, minimum=s['minimum'], restarts=1, iterations=s['warm_iterations'], seed=s['seed'] + 1 + r,
                            cap=max(30., cap - (time.monotonic() - started)), warm_start=ref['mapping'])
            seg, letters = warm['segmentation'], warm['recovered']
            rounds[-1].update(warm_log_likelihood=warm['log_likelihood'], warm_cap_hit=warm['cap_hit'])
    return dict(recovered=final, segmentation=seg, joint=dict(pieces=joint['pieces'], parses=joint['parses'], restart_scores=joint['restart_scores'],
                                                              restarts=joint['restarts'], cap_hit=joint['cap_hit'], seconds=joint['seconds']),
                rounds=rounds, seconds=time.monotonic() - started, case_cap_hit=time.monotonic() - started > cap)


def solve():
    frozen = verify()
    if (STATE / 'predictions.json').exists():
        raise FileExistsError('Predictions frozen')
    prior = CharacterPrior.load(STD / 'prior.npz')
    dev = read(ROOT / 'experiments/standard-decipherment/development.json')
    segmenter = Segmenter(read(STD / 'segmenter.json'), **dev['selected_segmenter']['parameters'])
    results = []; started = time.time()
    for i, case in enumerate(read(STATE / 'public.json')):
        checkpoint = STATE / 'partial' / f"{case['id']}.json"
        if checkpoint.exists():
            cached = read(checkpoint)
            if cached['freeze_sha256'] != digest(OUT / 'freeze.json') or cached['public_sha256'] != digest(STATE / 'public.json'):
                raise ValueError('Checkpoint provenance drift')
            results.append(cached['result']); continue
        result = decode(case['ciphertext'].split(), prior, frozen['settings'], frozen['case_cap_seconds'])
        result.update(id=case['id'], segmented=segmenter.segment(result['recovered']))
        write(checkpoint, dict(result=result, freeze_sha256=digest(OUT / 'freeze.json'), public_sha256=digest(STATE / 'public.json')))
        results.append(result)
        print(f"Ciphertext-only recovery {len(results)}/{len(read(STATE / 'public.json'))} ({time.time() - started:.1f}s this invocation)", flush=True)
    write(STATE / 'predictions.json', dict(results=results, code_sha256=hashes(), public_sha256=digest(STATE / 'public.json'),
                                           freeze_sha256=digest(OUT / 'freeze.json'), seconds_this_invocation=time.time() - started, at=time.time()))


def evaluate():
    verify(); challenge = read(STATE / 'challenge.json'); pred = read(STATE / 'predictions.json')
    for key, path in [('public_sha256', STATE / 'public.json'), ('answers_sha256', STATE / 'evaluator-only/answers.json'), ('freeze_sha256', OUT / 'freeze.json')]:
        if challenge[key] != digest(path):
            raise ValueError('Challenge drift')
    if pred['code_sha256'] != hashes() or any(pred[k] != challenge[k] for k in ('public_sha256', 'freeze_sha256')):
        raise ValueError('Prediction drift')
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    if {r['id'] for r in pred['results']} != set(answers):
        raise ValueError('Incomplete predictions')
    vendor = load_vendor(); glyphs = set(vendor.placeholder_to_glyph.values())
    previous = read(ROOT / 'experiments/standard-decipherment/results.json')['summary']['naibbe']
    rows = []
    for result in pred['results']:
        gold = answers[result['id']]; plain = gold['plaintext']; dense = plain.replace(' ', '')
        tokens = [c for c in (t for t in gold['trace'].split())]
        cipher_tokens = None
        # true segmentation from the trace, evaluator-side only
        public_tokens = next(c['ciphertext'] for c in read(STATE / 'public.json') if c['id'] == result['id']).split()
        truth = []
        for tok, chunk in zip(public_tokens, tokens):
            if len(chunk) == 1:
                truth.append((tok,))
            else:
                truth.append([(tok[:i], tok[i:]) for i in range(1, len(tok)) if tok[:i] in glyphs and tok[i:] in glyphs][0])
        agreement = sum(tuple(a) == tuple(b) for a, b in zip(result['segmentation'], truth)) / len(truth)
        ce = edit_distance(result['recovered'], dense); we = edit_distance(result['segmented'].split(), plain.split())
        rows.append(dict(id=result['id'], passage=gold['passage'], dataset=gold['dataset'], characters=len(dense), words=len(plain.split()),
                         character_errors=ce, word_errors=we, cer=ce / len(dense), wer=we / len(plain.split()),
                         gate=ce / len(dense) <= .01 and we / len(plain.split()) <= .1, segmentation_agreement=agreement,
                         recovered_letters=len(result['recovered']), joint=result['joint'], rounds=result['rounds'],
                         seconds=result['seconds'], case_cap_hit=result['case_cap_hit']))
    summary = {}
    for dataset in ('modern', 'historical'):
        subset = [r for r in rows if r['dataset'] == dataset]
        summary[dataset] = dict(cases=len(subset), cer=sum(r['character_errors'] for r in subset) / sum(r['characters'] for r in subset),
                                wer=sum(r['word_errors'] for r in subset) / sum(r['words'] for r in subset), gate_passes=sum(r['gate'] for r in subset),
                                previous_standard_method_cer=previous[dataset]['combined']['cer'], previous_letters='1,200-1,800 per passage')
    output = dict(summary=summary, cases=rows, challenge=challenge, predictions_sha256=digest(STATE / 'predictions.json'),
                  naibbe_gate=all(r['gate'] for r in rows), voynich_run=False, final_test_scored=False)
    write(OUT / 'results.json', output)
    print(json.dumps(dict(summary=summary, naibbe_gate=output['naibbe_gate']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'verify', 'prepare', 'solve', 'evaluate'])
    globals()[parser.parse_args().command]()
