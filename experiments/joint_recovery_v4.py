"""Round four: the round-two decoder, lexicon repair, refinement and polish on four new sealed passages.

Reuses the round-two and round-three stages (their modules stay frozen). Differences: the
lexicon repair between the second joint EM and the refinement, round three's passages added to
the exclusions, four Dante passages instead of two Dante and two modern (the ISDT test split
holds fewer than 1,000 fresh letters after three rounds; no new corpus was pinned), and settings
from experiments/joint-development-v4/results.json.

    python -m experiments.joint_recovery_v4 freeze | verify | prepare | solve | evaluate
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import time

from experiments.joint_development import load_vendor, role_units, majority_key
from experiments.joint_recovery import read, write
from experiments.segmentation import corpus
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.segmentation import Segmenter
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'artifacts/joint-recovery-v4'
OUT = ROOT / 'experiments/joint-recovery-v4'
DEV = ROOT / 'experiments/joint-development-v4'
STD = ROOT / 'artifacts/standard-decipherment'
PRIOR_PATH = ROOT / 'artifacts/verse-prior/prior-verse.npz'
CODE = ['voynich/joint_segments.py', 'voynich/joint_segments_v2.py', 'voynich/lexicon_repair.py', 'voynich/variable_units.py',
        'voynich/lexical_polish.py', 'voynich/homophonic.py', 'voynich/description_length.py', 'voynich/segmentation.py', 'voynich/decipher.py',
        'voynich/corpora.py', 'experiments/joint_development.py', 'experiments/joint_development_v2.py', 'experiments/joint_development_v3.py',
        'experiments/joint_development_v4.py', 'experiments/joint_recovery.py', 'experiments/joint_recovery_v4.py']
INPUTS = [DEV / 'results.json', DEV / 'PROTOCOL.md', STD / 'segmenter.json', ROOT / 'experiments/standard-decipherment/development.json',
          ROOT / 'experiments/decipherment-sources.json', ROOT / 'experiments/verse-prior/sources.json', PRIOR_PATH]
MIN_LETTERS, MAX_LETTERS = 5200, 6000
PASSAGES = 4  # all Dante; see the module docstring
CASE_CAP = 3600


def hashes(): return {p: digest(ROOT / p) for p in CODE}


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Already frozen')
    dev = read(DEV / 'results.json')
    minimum, passes = (int(part[-1]) for part in dev['selected'].split('_'))
    settings = dict(dev['settings'], repair_minimum=minimum, repair_passes=passes)
    write(OUT / 'freeze.json', dict(code_sha256=hashes(), inputs={str(p.relative_to(ROOT)): digest(p) for p in INPUTS},
                                    settings=settings, selected=dev['selected'], prior='prose+verse', prior_sha256=dev['prior_sha256'],
                                    min_letters=MIN_LETTERS, max_letters=MAX_LETTERS, case_cap_seconds=CASE_CAP, at=time.time()))
    print('Frozen; commit before prepare')


def verify(require_commit=True):
    frozen = read(OUT / 'freeze.json')
    if hashes() != frozen['code_sha256']:
        raise ValueError('Frozen code drift')
    for p, h in frozen['inputs'].items():
        if digest(ROOT / p) != h:
            raise ValueError('Frozen input drift: ' + p)
    if require_commit:
        tracked = [p for p in frozen['inputs'] if not p.startswith('artifacts/')]
        for p in CODE + ['experiments/joint-recovery-v4/freeze.json'] + tracked:
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != digest(ROOT / p):
                raise ValueError('Freeze not committed: ' + p)
    return frozen


def excluded_ids():
    excluded = {'modern': set(), 'historical': set()}
    for r in read(ROOT / 'artifacts/decipherment/evaluator-only/answers.json'):
        excluded['historical'].update(r['source_sentences'])
    for folder in ('segmentation', 'codebook-free', 'standard-decipherment', 'joint-recovery', 'joint-recovery-v2', 'joint-recovery-v3'):
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
    for dataset, rows, source in [('historical', *corpus('UD_Italian-Old', 'train'))]:
        # Walk the whole corpus in order and cut PASSAGES passages from consecutive fresh sentences.
        texts, ids, length, built = [], [], 0, 0
        for row in rows:
            text = normalize(' '.join(row['words'])); n = len(text.replace(' ', ''))
            if row['id'] in excluded[dataset] or text in seen or not text or length + n > MAX_LETTERS:
                continue
            texts.append(text); ids.append(row['id']); length += n
            if length >= MIN_LETTERS:
                excluded[dataset].update(ids)
                passages.append(dict(dataset=dataset, source=source, source_ids=ids, plaintext=' '.join(texts),
                                     passage=hashlib.sha256((source['path'] + ':'.join(ids)).encode()).hexdigest()[:16]))
                texts, ids, length, built = [], [], 0, built + 1
                if built == PASSAGES:
                    break
        if built < PASSAGES:
            raise ValueError(f'Not enough fresh {dataset} text for {PASSAGES} passages')
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
                                         cases=len(public), independent_passages=len(passages),
                                         letters=[len(p['plaintext'].replace(' ', '')) for p in passages], at=time.time()))
    print(f'Prepared {len(public)} Naibbe cases; references evaluator-only', flush=True)


def decode(tokens, prior, lexical, s, cap):
    """Round four: joint EM, usage pruning, joint EM, lexicon repair, refinement, lexical polish."""
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                      usage_floor=s['repair_usage_floor'], **em)
    units = role_units(repaired['segmentation'])
    ref = refine(units, prior, majority_key(units, repaired['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    pol = polish(units, prior, ref['mapping'], lexical, weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'],
                 sweeps=s['polish_sweeps'], cap=min(s['polish_cap'], max(30., cap - (time.monotonic() - started))))
    return dict(recovered=pol['recovered'], refined_recovered=ref['recovered'], segmentation=repaired['segmentation'],
                unrepaired_segmentation=second['segmentation'],
                joint=dict(first_pieces=len(first['candidate_pieces']), pruned_pieces=len(second['candidate_pieces']),
                           repaired_pieces=len(repaired['candidate_pieces']), repair=repaired['repair'],
                           first_restart_scores=first['restart_scores'], second_restart_scores=second['restart_scores'],
                           repaired_restart_scores=repaired['restart_scores'],
                           first_cap_hit=first['cap_hit'], second_cap_hit=second['cap_hit'], repaired_cap_hit=repaired['cap_hit']),
                refined_bits=ref['search_bits'], refine_cap_hit=ref['cap_hit'],
                polish=dict(changes=len(pol['changes']), sweeps=pol['sweeps'], seconds=pol['seconds'], cap_hit=pol['cap_hit'], record=pol['changes']),
                seconds=time.monotonic() - started, case_cap_hit=time.monotonic() - started > cap)


def solve():
    frozen = verify()
    if (STATE / 'predictions.json').exists():
        raise FileExistsError('Predictions frozen')
    prior = CharacterPrior.load(PRIOR_PATH)
    dev = read(ROOT / 'experiments/standard-decipherment/development.json')
    segmenter = Segmenter(read(STD / 'segmenter.json'), **dev['selected_segmenter']['parameters'])
    lexical = LexicalCost(segmenter)
    results = []; started = time.time()
    cases = read(STATE / 'public.json')
    for case in cases:
        checkpoint = STATE / 'partial' / f"{case['id']}.json"
        if checkpoint.exists():
            cached = read(checkpoint)
            if cached['freeze_sha256'] != digest(OUT / 'freeze.json') or cached['public_sha256'] != digest(STATE / 'public.json'):
                raise ValueError('Checkpoint provenance drift')
            results.append(cached['result']); continue
        result = decode(case['ciphertext'].split(), prior, lexical, frozen['settings'], frozen['case_cap_seconds'])
        result.update(id=case['id'], segmented=segmenter.segment(result['recovered']), refined_segmented=segmenter.segment(result['refined_recovered']))
        write(checkpoint, dict(result=result, freeze_sha256=digest(OUT / 'freeze.json'), public_sha256=digest(STATE / 'public.json')))
        results.append(result)
        print(f"Ciphertext-only recovery {len(results)}/{len(cases)} ({time.time() - started:.1f}s this invocation)", flush=True)
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
    public = {c['id']: c['ciphertext'].split() for c in read(STATE / 'public.json')}
    if {r['id'] for r in pred['results']} != set(answers):
        raise ValueError('Incomplete predictions')
    vendor = load_vendor(); glyphs = set(vendor.placeholder_to_glyph.values())
    previous = {name: read(ROOT / f'experiments/{folder}/results.json')['summary']
                for name, folder in (('round_one', 'joint-recovery'), ('round_two', 'joint-recovery-v2'), ('round_three', 'joint-recovery-v3'))}
    rows = []
    for result in pred['results']:
        gold = answers[result['id']]; plain = gold['plaintext']; dense = plain.replace(' ', '')
        truth = []
        for tok, chunk in zip(public[result['id']], gold['trace'].split()):
            truth.append((tok,) if len(chunk) == 1 else
                         [(tok[:i], tok[i:]) for i in range(1, len(tok)) if tok[:i] in glyphs and tok[i:] in glyphs][0])
        true_pieces = {p for t in truth for p in t}
        agreement = lambda seg: sum(tuple(a) == tuple(b) for a, b in zip(seg, truth)) / len(truth)
        # error attribution, evaluator-side
        pos_r = pos_g = 0; ok_letters = wrong_in_ok = bad_letters = 0
        for parts, tp in zip(result['segmentation'], truth):
            lr, lg = len(parts), len(tp)
            if tuple(parts) == tuple(tp):
                ok_letters += lg; wrong_in_ok += sum(a != b for a, b in zip(result['recovered'][pos_r:pos_r + lr], dense[pos_g:pos_g + lg]))
            else:
                bad_letters += lg
            pos_r += lr; pos_g += lg
        def metrics(text, segmented):
            ce = edit_distance(text, dense); we = edit_distance(segmented.split(), plain.split())
            return dict(character_errors=ce, word_errors=we, cer=ce / len(dense), wer=we / len(plain.split()),
                        gate=ce / len(dense) <= .01 and we / len(plain.split()) <= .1)
        rows.append(dict(id=result['id'], passage=gold['passage'], dataset=gold['dataset'], characters=len(dense), words=len(plain.split()),
                         polished=metrics(result['recovered'], result['segmented']), refined_only=metrics(result['refined_recovered'], result['refined_segmented']),
                         segmentation_agreement=agreement(result['segmentation']), unrepaired_segmentation_agreement=agreement(result['unrepaired_segmentation']),
                         true_pieces=len(true_pieces), error_rate_in_correct_tokens=wrong_in_ok / max(1, ok_letters),
                         share_of_letters_misparsed=bad_letters / len(dense), recovered_letters=len(result['recovered']),
                         joint=result['joint'], polish_changes=result['polish']['changes'], polish_cap_hit=result['polish']['cap_hit'],
                         refine_cap_hit=result['refine_cap_hit'], seconds=result['seconds'], case_cap_hit=result['case_cap_hit']))
    summary = {}
    for dataset in ('historical',):
        subset = [r for r in rows if r['dataset'] == dataset]
        summary[dataset] = {}
        for stage in ('polished', 'refined_only'):
            summary[dataset][stage] = dict(cases=len(subset), cer=sum(r[stage]['character_errors'] for r in subset) / sum(r['characters'] for r in subset),
                                           wer=sum(r[stage]['word_errors'] for r in subset) / sum(r['words'] for r in subset),
                                           gate_passes=sum(r[stage]['gate'] for r in subset))
        for name, prev in previous.items():
            block = prev[dataset]
            summary[dataset][f'{name}_cer'] = block['polished']['cer'] if 'polished' in block else block['cer']
            summary[dataset][f'{name}_wer'] = block['polished']['wer'] if 'polished' in block else block['wer']
    output = dict(summary=summary, cases=rows, challenge=challenge, predictions_sha256=digest(STATE / 'predictions.json'),
                  naibbe_gate=all(r['polished']['gate'] for r in rows), voynich_run=False, final_test_scored=False,
                  modern_passages='none: the ISDT test split holds fewer than 1,000 fresh letters after three rounds')
    write(OUT / 'results.json', output)
    print(json.dumps(dict(summary=summary, naibbe_gate=output['naibbe_gate']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'verify', 'prepare', 'solve', 'evaluate'])
    globals()[parser.parse_args().command]()
