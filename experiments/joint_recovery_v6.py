"""Round six: sealed 20,800-letter Naibbe cases; square-root baseline (S) vs context reparse (R), paired.

    python -m experiments.joint_recovery_v6 freeze | fetch | prepare | solve <index> | collect | evaluate
See experiments/joint-recovery-v6/PROTOCOL.md.
"""
import argparse
import hashlib
import io
import random
import subprocess
import time

from experiments.joint_development import load_vendor, role_units, majority_key
from experiments.joint_recovery import read, write
from experiments.joint_recovery_v4 import ROOT, excluded_ids, verify as round_four
from experiments.length_scaling_v2 import scaled
from experiments.length_scaling_v3 import REPARSE_ROUNDS, REPARSE_WIDTH
from experiments.segmentation import corpus
from experiments.word_segmentation_fresh import fetch as fetch_file, seen_ngrams
from experiments.word_segmentation_v3 import verify as v3_frozen
from experiments.word_segmentation_v3_fresh import historical_rows, released_ngrams, solvers
from voynich.context_reparse import reparse
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexical_polish import LexicalCost, polish
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

OUT = ROOT / 'experiments/joint-recovery-v6'
STATE = ROOT / 'artifacts/joint-recovery-v6'
PRIOR_PATH = ROOT / 'artifacts/verse-prior/prior-verse.npz'
MIN_LETTERS, MAX_LETTERS, PER_SOURCE = 20800, 22000, 2
PARTUT_REVISION = '6ae975aeb16378d9aaf102ecccaf1a083e636b44'
PARTUT_TRAIN = STATE / 'sources/it_partut-ud-train.conllu'
CODE = ['experiments/joint_recovery_v6.py', 'experiments/length_scaling_v2.py', 'experiments/length_scaling_v3.py',
        'voynich/context_reparse.py', 'voynich/lexicon_repair.py', 'voynich/lexical_polish.py', 'voynich/unknown_words.py',
        'experiments/joint_recovery_v4.py', 'experiments/word_segmentation_v3_fresh.py']
INPUTS = ['experiments/joint-recovery-v6/PROTOCOL.md', 'experiments/joint-recovery-v4/freeze.json',
          'experiments/word-segmentation-v3/freeze.json', 'experiments/length-scaling-v3/summary.json']
RELEASED = ['experiments/word-segmentation-v3-fresh/evaluated-records.json', 'experiments/joint-recovery-v5/evaluated-records.json',
            'experiments/language-id/evaluated-records.json']


def freeze():
    if (OUT / 'freeze.json').exists(): raise FileExistsError('Already frozen')
    four = round_four(); v3 = v3_frozen()
    write(OUT / 'freeze.json', dict(code_sha256={p: digest(ROOT / p) for p in CODE}, inputs={p: digest(ROOT / p) for p in INPUTS},
          base_settings=four['settings'], scaling='sqrt', letters=MIN_LETTERS, settings=scaled(four['settings'], MIN_LETTERS, 'sqrt'),
          reparse=dict(rounds=REPARSE_ROUNDS, width=REPARSE_WIDTH), segmenter_v3=v3['selected'], partut_revision=PARTUT_REVISION, at=time.time()))
    print('Frozen; commit before fetch')


def verify():
    f = read(OUT / 'freeze.json'); round_four(); v3_frozen()
    for p, h in list(f['code_sha256'].items()) + list(f['inputs'].items()):
        if digest(ROOT / p) != h: raise ValueError('Frozen drift: ' + p)
    for p in list(f['code_sha256']) + INPUTS + ['experiments/joint-recovery-v6/freeze.json']:
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)).hexdigest() != digest(ROOT / p):
            raise ValueError('Freeze not committed: ' + p)
    return f


def fetch():
    verify()
    if (OUT / 'sources.json').exists(): raise FileExistsError('Sources pinned')
    r = fetch_file(f'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParTUT/{PARTUT_REVISION}/it_partut-ud-train.conllu', PARTUT_TRAIN)
    r.update(revision=PARTUT_REVISION, repository='UD_Italian-ParTUT', license='CC BY-NC-SA 4.0')
    write(OUT / 'sources.json', dict(files=[r]))
    print(r)


def released():
    ids, grams = set(), set()
    for path in RELEASED:
        data = read(ROOT / path); rows = data['answers']
        for r in rows:
            ids.update(i.split(':part')[0] for i in (r.get('source_ids') or []))
            w = r['plaintext'].split(); grams.update(tuple(w[i:i + 20]) for i in range(len(w) - 19))
    return ids, grams


def pack(rows, seen, dataset, source):
    out, texts, ids, size = [], [], [], 0
    for row in rows:
        w = row['text'].split(); n = len(row['text'].replace(' ', ''))
        if not n or any(tuple(w[i:i + 20]) in seen for i in range(len(w) - 19)): continue
        if size + n > MAX_LETTERS:
            texts, ids, size = [], [], 0
            if n > MAX_LETTERS: continue
        texts.append(row['text']); ids.append(row['id']); size += n
        if size >= MIN_LETTERS:
            out.append(dict(dataset=dataset, source=source, source_ids=ids, plaintext=' '.join(texts)))
            texts, ids, size = [], [], 0
            if len(out) == PER_SOURCE: return out
    raise ValueError(f'Not enough fresh {dataset} text')


def prepare():
    verify()
    if (STATE / 'public.json').exists(): raise FileExistsError('Already prepared')
    for r in read(OUT / 'sources.json')['files']:
        if digest(ROOT / r['path']) != r['sha256']: raise ValueError('Source drift')
    ids, grams = released(); seen = seen_ngrams() | released_ngrams() | grams
    excluded = excluded_ids()
    for folder in ('joint-recovery-v4', 'joint-recovery-v5'):
        for r in read(ROOT / f'artifacts/{folder}/evaluator-only/answers.json'): excluded['historical'].update(r['source_ids'])
    isdt = set()
    for split in ('train', 'dev'):
        rows, _ = corpus('UD_Italian-ISDT', split); isdt.update(normalize(' '.join(r['words'])) for r in rows)
    old, source = corpus('UD_Italian-Old', 'train')
    dante = [dict(id=r['id'], text=normalize(' '.join(r['words']))) for r in old]
    dante = [r for r in dante if r['id'] not in excluded['historical'] and r['text'] not in isdt]
    compagni = [r for r in historical_rows()[0] if r['id'] not in ids]
    modern = [dict(id='partut-train:' + r['id'], text=normalize(' '.join(r['words']))) for r in conllu_sentences(PARTUT_TRAIN)]
    passages = (pack(dante, seen, 'dante', source['path']) + pack(compagni, seen, 'compagni', 'compagni')
                + pack(modern, seen, 'modern', str(PARTUT_TRAIN.relative_to(ROOT))))
    vendor = load_vendor(); public, answers = [], []
    for passage in passages:
        dense = passage['plaintext'].replace(' ', '')
        seed = random.SystemRandom().randrange(2 ** 63); rng = random.Random(seed)
        alphabet = list(ALPHABET); rng.shuffle(alphabet); key = dict(zip(ALPHABET, alphabet)); inverse = {v: k for k, v in key.items()}
        random.seed(seed); trace = io.StringIO()
        tokens = vendor.encrypt_naibbe(''.join(key[c] for c in dense), vendor.naibbe_tables, vendor.placeholder_to_glyph, use_78=False, pre_plaintext_file=trace)
        if ''.join(inverse[c] for c in ''.join(trace.getvalue().split())) != dense: raise ValueError('Roundtrip failed')
        cipher = ' '.join(tokens); ident = hashlib.sha256((cipher + str(seed)).encode()).hexdigest()[:16]
        public.append(dict(id=ident, ciphertext=cipher))
        answers.append(dict(passage, id=ident, family='naibbe', seed=seed, trace=trace.getvalue(), roundtrip=True))
    private = STATE / 'evaluator-only'; private.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
          freeze_sha256=digest(OUT / 'freeze.json'), sources_sha256=digest(OUT / 'sources.json'),
          head_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          cases=len(public), letters=[len(a['plaintext'].replace(' ', '')) for a in answers], at=time.time()))
    print(f'Prepared {len(public)} cases; references evaluator-only', flush=True)


def solve(index):
    f = verify(); s = f['settings']
    case = read(STATE / 'public.json')[index]
    checkpoint = STATE / 'partial' / f"{case['id']}.json"
    if checkpoint.exists(): raise FileExistsError('Case solved')
    prior = CharacterPrior.load(PRIOR_PATH); segmenters = solvers(v3_frozen()); lexical = LexicalCost(segmenters['baseline'])
    tokens = case['ciphertext'].split(); cap = 3600 * f['letters'] / 5200; started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    rep = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                 usage_floor=s['repair_usage_floor'], **em)
    units = role_units(rep['segmentation'])
    ref = refine(units, prior, majority_key(units, rep['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    shared = time.monotonic() - started
    args = dict(weight=s['polish_weight'], radius=s['polish_radius'], shortlist=s['polish_shortlist'], sweeps=s['polish_sweeps'])
    pol_s = polish(units, prior, ref['mapping'], lexical, **args, cap=s['polish_cap'])
    segmentation, r_units, r_ref, rounds = rep['segmentation'], units, ref, []
    for r in range(f['reparse']['rounds']):
        rp = reparse(tokens, r_ref['mapping'], prior, width=f['reparse']['width'])
        changed = sum(tuple(a) != tuple(b) for a, b in zip(rp['segmentation'], segmentation))
        segmentation = rp['segmentation']; r_units = role_units(segmentation)
        r_ref = refine(r_units, prior, majority_key(r_units, rp['recovered']), (), seed=r + 2, kicks=s['refine_kicks'],
                       kick_size=s['refine_kick_size'], cap=s['refine_cap'])
        rounds.append(dict(changed_parses=changed, seconds=rp['seconds'], refine_cap_hit=r_ref['cap_hit']))
    pol_r = polish(r_units, prior, r_ref['mapping'], lexical, **args, cap=s['polish_cap'])
    result = dict(id=case['id'], S=dict(recovered=pol_s['recovered'], segmented=segmenters['v3'].segment(pol_s['recovered']), cap_hit=pol_s['cap_hit']),
                  R=dict(recovered=pol_r['recovered'], segmented=segmenters['v3'].segment(pol_r['recovered']), cap_hit=pol_r['cap_hit']),
                  S_segmentation=rep['segmentation'], R_segmentation=segmentation, reparse_rounds=rounds,
                  cap_hits=dict(first=first['cap_hit'], second=second['cap_hit'], repair=rep['cap_hit'], refine=ref['cap_hit']),
                  shared_seconds=shared, seconds=time.monotonic() - started)
    write(checkpoint, dict(result=result, freeze_sha256=digest(OUT / 'freeze.json'), public_sha256=digest(STATE / 'public.json')))
    print(f"Round six case {index}: {result['seconds']:.0f}s", flush=True)


def collect():
    verify(); cases = read(STATE / 'public.json'); results = []
    for case in cases:
        cached = read(STATE / 'partial' / f"{case['id']}.json")
        if cached['freeze_sha256'] != digest(OUT / 'freeze.json') or cached['public_sha256'] != digest(STATE / 'public.json'):
            raise ValueError('Checkpoint provenance drift')
        results.append(cached['result'])
    write(STATE / 'predictions.json', dict(results=results, public_sha256=digest(STATE / 'public.json'), freeze_sha256=digest(OUT / 'freeze.json'), at=time.time()))


def evaluate():
    verify(); c = read(STATE / 'challenge.json'); pred = read(STATE / 'predictions.json')
    for key, path in [('public_sha256', STATE / 'public.json'), ('answers_sha256', STATE / 'evaluator-only/answers.json'), ('freeze_sha256', OUT / 'freeze.json')]:
        if c[key] != digest(path): raise ValueError('Challenge drift: ' + key)
    answers = {r['id']: r for r in read(STATE / 'evaluator-only/answers.json')}
    public = {r['id']: r['ciphertext'].split() for r in read(STATE / 'public.json')}
    glyphs = set(load_vendor().placeholder_to_glyph.values()); rows = []
    for r in pred['results']:
        gold = answers[r['id']]; plain = gold['plaintext']; dense = plain.replace(' ', ''); words = plain.split()
        truth = [(tok,) if len(ch) == 1 else [(tok[:i], tok[i:]) for i in range(1, len(tok)) if tok[:i] in glyphs and tok[i:] in glyphs][0]
                 for tok, ch in zip(public[r['id']], gold['trace'].split())]
        def m(arm, seg):
            ce, we = edit_distance(r[arm]['recovered'], dense), edit_distance(r[arm]['segmented'].split(), words)
            return dict(character_errors=ce, word_errors=we, cer=ce / len(dense), wer=we / len(words), gate=ce / len(dense) <= .01 and we / len(words) <= .1,
                        agreement=sum(tuple(a) == tuple(b) for a, b in zip(seg, truth)) / len(truth), polish_cap_hit=r[arm]['cap_hit'])
        rows.append(dict(id=r['id'], dataset=gold['dataset'], characters=len(dense), words=len(words), S=m('S', r['S_segmentation']),
                         R=m('R', r['R_segmentation']), reparse_rounds=r['reparse_rounds'], cap_hits=r['cap_hits'], seconds=r['seconds']))
    def pool(sub, arm):
        return dict(cases=len(sub), cer=sum(x[arm]['character_errors'] for x in sub) / sum(x['characters'] for x in sub),
                    wer=sum(x[arm]['word_errors'] for x in sub) / sum(x['words'] for x in sub), gate_passes=sum(x[arm]['gate'] for x in sub))
    summary = {name: {arm: pool(sub, arm) for arm in 'SR'} for name, sub in
               [('all', rows)] + [(d, [x for x in rows if x['dataset'] == d]) for d in ('dante', 'compagni', 'modern')]}
    primary = summary['all']['S']['cer'] - summary['all']['R']['cer'] >= .005
    out = dict(summary=summary, cases=rows, primary_passed=primary, naibbe_gate=all(x['R']['gate'] for x in rows),
               challenge=c, predictions_sha256=digest(STATE / 'predictions.json'), voynich_run=False)
    write(OUT / 'results.json', out)
    write(OUT / 'evaluated-records.json', dict(public=read(STATE / 'public.json'), answers=list(answers.values()), predictions=pred))
    for name, s in summary.items():
        print(name, {arm: (round(100 * v['cer'], 2), round(100 * v['wer'], 2), v['gate_passes']) for arm, v in s.items()})
    print('primary_passed', primary, 'naibbe_gate', out['naibbe_gate'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['freeze', 'fetch', 'prepare', 'solve', 'collect', 'evaluate'])
    p.add_argument('index', nargs='?', type=int)
    a = p.parse_args()
    solve(a.index) if a.command == 'solve' else globals()[a.command]()
