"""Round five: round-four Naibbe decoding, paired arms that change only the segmenter.

A = round four exactly; B = A's letters with the v3 segmenter; C = polish and segment with v3.
Shared stages run once per case. Eight sealed cases: four Dante, four unused Compagni.
See experiments/joint-recovery-v5/PROTOCOL.md.

    python -m experiments.joint_recovery_v5 freeze | verify | prepare | solve | evaluate
"""
import argparse
import hashlib
import io
import random
import subprocess
import time

from experiments.joint_development import load_vendor, role_units, majority_key
from experiments.joint_recovery import read, write
from experiments.joint_recovery_v4 import (ROOT, STD, PRIOR_PATH, MIN_LETTERS, MAX_LETTERS, CASE_CAP,
                                           excluded_ids, verify as round_four)
from experiments.segmentation import corpus
from experiments.word_segmentation_fresh import passages as pack, seen_ngrams
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import historical_rows, released_ngrams, solvers
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexical_polish_v3 import SpellingLexicalCost
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

STATE = ROOT / 'artifacts/joint-recovery-v5'
OUT = ROOT / 'experiments/joint-recovery-v5'
CODE = ['experiments/joint_recovery_v5.py', 'voynich/lexical_polish_v3.py', 'voynich/unknown_words.py',
        'experiments/word_segmentation_v3.py', 'experiments/word_segmentation_v3_fresh.py', 'experiments/joint_recovery_v4.py',
        'voynich/lexical_polish.py', 'voynich/segmentation.py']
INPUTS = ['experiments/joint-recovery-v5/PROTOCOL.md', 'experiments/joint-recovery-v4/freeze.json',
          'experiments/word-segmentation-v3/freeze.json', 'experiments/word-segmentation-v3-fresh/evaluated-records.json',
          'experiments/word-segmentation-v3-fresh/sources.json']
PER_SOURCE = 4
V3_RELEASED = 'experiments/word-segmentation-v3-fresh/evaluated-records.json'


def freeze():
    if (OUT / 'freeze.json').exists(): raise FileExistsError('Already frozen')
    four, v3 = round_four(), v3_frozen()
    write(OUT / 'freeze.json', dict(code_sha256={p: digest(ROOT / p) for p in CODE}, inputs={p: digest(ROOT / p) for p in INPUTS},
                                    settings=four['settings'], segmenter_v3=v3['selected'], case_cap_seconds=CASE_CAP, at=time.time()))
    print('Frozen; commit before prepare')


def verify():
    frozen = read(OUT / 'freeze.json'); round_four(); v3_frozen()
    for p, h in list(frozen['code_sha256'].items()) + list(frozen['inputs'].items()):
        if digest(ROOT / p) != h: raise ValueError('Frozen drift: ' + p)
    for p in list(frozen['code_sha256']) + INPUTS + ['experiments/joint-recovery-v5/freeze.json']:
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)).hexdigest() != digest(ROOT / p):
            raise ValueError('Freeze not committed: ' + p)
    return frozen


def dante_passages():
    """Round four's walk over UD_Italian-Old, excluding every earlier round including round four."""
    excluded = excluded_ids()
    for r in read(ROOT / 'artifacts/joint-recovery-v4/evaluator-only/answers.json'):
        excluded[r['dataset']].update(r['source_ids'])
    seen = set()
    for split in ('train', 'dev'):
        rows, _ = corpus('UD_Italian-ISDT', split)
        seen.update(normalize(' '.join(r['words'])) for r in rows)
    rows, source = corpus('UD_Italian-Old', 'train')
    out, texts, ids, length = [], [], [], 0
    for row in rows:
        text = normalize(' '.join(row['words'])); n = len(text.replace(' ', ''))
        if row['id'] in excluded['historical'] or text in seen or not text or length + n > MAX_LETTERS:
            continue
        texts.append(text); ids.append(row['id']); length += n
        if length >= MIN_LETTERS:
            out.append(dict(dataset='dante', source=source['path'], source_ids=ids, plaintext=' '.join(texts)))
            texts, ids, length = [], [], 0
            if len(out) == PER_SOURCE: return out
    raise ValueError('Not enough fresh Dante')


def compagni_passages():
    released = {i.split(':part')[0] for r in read(ROOT / V3_RELEASED)['answers'] for i in r['source_ids']}
    rows = [r for r in historical_rows()[0] if r['id'] not in released]
    seen = seen_ngrams() | released_ngrams()
    for r in read(ROOT / V3_RELEASED)['answers']:
        w = r['plaintext'].split(); seen.update(tuple(w[i:i + 20]) for i in range(len(w) - 19))
    blocks, rejected = pack(rows, seen)
    return [dict(dataset='compagni', source='compagni', **b) for b in blocks], rejected, len(released)


def prepare():
    verify()
    if (STATE / 'public.json').exists(): raise FileExistsError('Already prepared')
    compagni, rejected, released = compagni_passages()
    vendor = load_vendor(); public, answers = [], []
    for passage in dante_passages() + compagni:
        dense = passage['plaintext'].replace(' ', '')
        seed = random.SystemRandom().randrange(2 ** 63); rng = random.Random(seed)
        alphabet = list(ALPHABET); rng.shuffle(alphabet); key = dict(zip(ALPHABET, alphabet)); inverse = {v: k for k, v in key.items()}
        coded = ''.join(key[c] for c in dense)
        random.seed(seed); trace = io.StringIO()
        tokens = vendor.encrypt_naibbe(coded, vendor.naibbe_tables, vendor.placeholder_to_glyph, use_78=False, pre_plaintext_file=trace)
        if ''.join(inverse[c] for c in ''.join(trace.getvalue().split())) != dense: raise ValueError('Roundtrip failed')
        cipher = ' '.join(tokens); ident = hashlib.sha256((cipher + str(seed)).encode()).hexdigest()[:16]
        public.append(dict(id=ident, ciphertext=cipher))
        answers.append(dict(passage, id=ident, family='naibbe', seed=seed, trace=trace.getvalue(), roundtrip=True))
    private = STATE / 'evaluator-only'; private.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
          freeze_sha256=digest(OUT / 'freeze.json'), head_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          cases=len(public), compagni_rejected_ids=rejected, compagni_released_paragraphs=released,
          letters=[len(a['plaintext'].replace(' ', '')) for a in answers], at=time.time()))
    print(f'Prepared {len(public)} Naibbe cases; references evaluator-only', flush=True)


def decode(tokens, prior, lexical_a, lexical_c, s, cap):
    """Round four's decode (arm A, unchanged), plus arm C's polish from the same refined key."""
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                      usage_floor=s['repair_usage_floor'], **em)
    units = role_units(repaired['segmentation'])
    ref = refine(units, prior, majority_key(units, repaired['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    polish_args = dict(weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'], sweeps=s['polish_sweeps'])
    shared = time.monotonic() - started
    a = polish(units, prior, ref['mapping'], lexical_a, **polish_args, cap=min(s['polish_cap'], max(30., cap - shared)))
    c = polish(units, prior, ref['mapping'], lexical_c, **polish_args, cap=min(s['polish_cap'], max(30., cap - shared)))
    arm = lambda p: dict(recovered=p['recovered'], changes=len(p['changes']), sweeps=p['sweeps'], seconds=p['seconds'], cap_hit=p['cap_hit'])
    return dict(a=arm(a), c=arm(c), refined_recovered=ref['recovered'], segmentation=repaired['segmentation'],
                refine_cap_hit=ref['cap_hit'], shared_seconds=shared, seconds=time.monotonic() - started)


def solve():
    frozen = verify()
    if (STATE / 'predictions.json').exists(): raise FileExistsError('Predictions frozen')
    prior = CharacterPrior.load(PRIOR_PATH)
    segmenters = solvers(v3_frozen())
    lexical_a, lexical_c = LexicalCost(segmenters['baseline']), SpellingLexicalCost(segmenters['v3'])
    cases = read(STATE / 'public.json'); results = []; started = time.time()
    for case in cases:
        checkpoint = STATE / 'partial' / f"{case['id']}.json"
        if checkpoint.exists():
            cached = read(checkpoint)
            if cached['freeze_sha256'] != digest(OUT / 'freeze.json') or cached['public_sha256'] != digest(STATE / 'public.json'):
                raise ValueError('Checkpoint provenance drift')
            results.append(cached['result']); continue
        r = decode(case['ciphertext'].split(), prior, lexical_a, lexical_c, frozen['settings'], frozen['case_cap_seconds'])
        r.update(id=case['id'], a_segmented=segmenters['baseline'].segment(r['a']['recovered']),
                 b_segmented=segmenters['v3'].segment(r['a']['recovered']), c_segmented=segmenters['v3'].segment(r['c']['recovered']))
        write(checkpoint, dict(result=r, freeze_sha256=digest(OUT / 'freeze.json'), public_sha256=digest(STATE / 'public.json')))
        results.append(r)
        print(f"Round five {len(results)}/{len(cases)}: {r['seconds']:.0f}s case, polish A {r['a']['seconds']:.0f}s, C {r['c']['seconds']:.0f}s "
              f"({time.time() - started:.0f}s this invocation)", flush=True)
    write(STATE / 'predictions.json', dict(results=results, public_sha256=digest(STATE / 'public.json'),
                                           freeze_sha256=digest(OUT / 'freeze.json'), at=time.time()))


def evaluate():
    verify(); c = read(STATE / 'challenge.json'); pred = read(STATE / 'predictions.json')
    for key, path in [('public_sha256', STATE / 'public.json'), ('answers_sha256', STATE / 'evaluator-only/answers.json'), ('freeze_sha256', OUT / 'freeze.json')]:
        if c[key] != digest(path): raise ValueError('Challenge drift: ' + key)
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    if {r['id'] for r in pred['results']} != set(answers): raise ValueError('Incomplete predictions')
    rows = []
    for r in pred['results']:
        plain = answers[r['id']]['plaintext']; dense = plain.replace(' ', ''); words = plain.split()
        def m(text, segmented):
            ce, we = edit_distance(text, dense), edit_distance(segmented.split(), words)
            return dict(character_errors=ce, word_errors=we, cer=ce / len(dense), wer=we / len(words), gate=ce / len(dense) <= .01 and we / len(words) <= .1)
        rows.append(dict(id=r['id'], dataset=answers[r['id']]['dataset'], characters=len(dense), words=len(words),
                         A=m(r['a']['recovered'], r['a_segmented']), B=m(r['a']['recovered'], r['b_segmented']),
                         C=m(r['c']['recovered'], r['c_segmented']), polish_cap_hit=dict(A=r['a']['cap_hit'], C=r['c']['cap_hit']),
                         polish_changes=dict(A=r['a']['changes'], C=r['c']['changes']), refine_cap_hit=r['refine_cap_hit'], seconds=r['seconds']))
    def pool(sub, arm):
        return dict(cases=len(sub), cer=sum(x[arm]['character_errors'] for x in sub) / sum(x['characters'] for x in sub),
                    wer=sum(x[arm]['word_errors'] for x in sub) / sum(x['words'] for x in sub), gate_passes=sum(x[arm]['gate'] for x in sub))
    summary = {name: {arm: pool(sub, arm) for arm in 'ABC'} for name, sub in
               [('all', rows), ('dante', [x for x in rows if x['dataset'] == 'dante']), ('compagni', [x for x in rows if x['dataset'] == 'compagni'])]}
    a, cc = summary['all']['A'], summary['all']['C']
    primary = a['wer'] - cc['wer'] >= .03 and cc['cer'] - a['cer'] <= .005
    out = dict(summary=summary, cases=rows, primary_passed=primary, naibbe_gate=all(x['C']['gate'] for x in rows),
               challenge=c, predictions_sha256=digest(STATE / 'predictions.json'), voynich_run=False)
    write(OUT / 'results.json', out)
    write(OUT / 'evaluated-records.json', dict(public=read(STATE / 'public.json'), answers=list(answers.values()), predictions=pred))
    for name, s in summary.items():
        print(name, {arm: (round(100 * v['cer'], 2), round(100 * v['wer'], 2), v['gate_passes']) for arm, v in s.items()})
    print('primary_passed', primary, 'naibbe_gate', out['naibbe_gate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'verify', 'prepare', 'solve', 'evaluate'])
    command = parser.parse_args().command
    print(globals()[command]())
