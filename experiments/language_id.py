"""Language-identification control: decode Naibbe ciphertext of five known languages under five priors.

    python -m experiments.language_id freeze | prepare | solve | evaluate

See experiments/language-id/PROTOCOL.md. No download; corpora are already pinned.
"""
import argparse
import hashlib
import io
import random
import subprocess
import time

from experiments.joint_development import load_vendor, role_units, majority_key
from experiments.joint_recovery import read, write
from experiments.joint_recovery_v4 import ROOT, verify as round_four
from experiments.historical_sources import verify as historical
from experiments.segmentation import corpus
from experiments.word_segmentation_fresh import passages as pack
from experiments.word_segmentation_v3_fresh import historical_rows
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexicon_repair import repair
from voynich.variable_units import refine

OUT = ROOT / 'experiments/language-id'
STATE = ROOT / 'artifacts/language-id'
SOURCES = ROOT / 'artifacts/language-sources'
FILES = dict(latin='UD_Latin-ITTB/la_ittb-ud-train.conllu', old_french='UD_Old_French-PROFITEROLE/fro_profiterole-ud-train.conllu',
             german='UD_German-GSD/de_gsd-ud-train.conllu', english='UD_English-EWT/en_ewt-ud-train.conllu')
LANGUAGES = ('latin', 'old_french', 'german', 'english', 'italian')
MIN_LETTERS, MAX_LETTERS, HELD_OUT = 5200, 6000, 20000
CASE_CAP = 3600  # as round four
RELEASED = ['experiments/word-segmentation-v3-fresh/evaluated-records.json', 'artifacts/joint-recovery-v5/evaluator-only/answers.json']


def letters(text): return len(text.replace(' ', ''))


def take(sentences, budget):
    out, n = [], 0
    for s in sentences:
        if n >= budget: break
        out.append(s); n += letters(s)
    return out


def split_tail(sentences):
    """From the end: a passage of MIN..MAX letters, then HELD_OUT letters; the rest is prior text."""
    passage, n, i = [], 0, len(sentences)
    while n < MIN_LETTERS:
        i -= 1
        if n + letters(sentences[i]) > MAX_LETTERS: continue
        passage.insert(0, sentences[i]); n += letters(sentences[i])
    held, m = [], 0
    while m < HELD_OUT:
        i -= 1; held.insert(0, sentences[i]); m += letters(sentences[i])
    return sentences[:i], held, ' '.join(passage)


def texts():
    """Per language: prior sentences (uncapped), held-out sentences, passage plaintext."""
    out = {}
    for lang, path in FILES.items():
        sentences = [t for t in (normalize(' '.join(r['words'])) for r in conllu_sentences(SOURCES / path)) if t]
        out[lang] = split_tail(sentences)
    hist = [normalize(' '.join(r['paragraphs'])) for r in historical() if r['split'] == 'train']
    isdt_train, _ = corpus('UD_Italian-ISDT', 'train'); isdt_dev, _ = corpus('UD_Italian-ISDT', 'dev')
    modern = [normalize(' '.join(r['words'])) for r in isdt_train]
    held = take([normalize(' '.join(r['words'])) for r in isdt_dev], HELD_OUT)
    out['italian'] = (hist + modern, held, None)
    return out


def budget(t): return min(sum(map(letters, prior)) for prior, _, _ in t.values())


def fit_priors(t, n):
    return {lang: CharacterPrior.fit(take(prior, n)) for lang, (prior, _, _) in t.items()}


def freeze():
    if (OUT / 'freeze.json').exists(): raise FileExistsError('Already frozen')
    four = round_four(); t = texts(); n = budget(t)
    priors = fit_priors(t, n)
    entropy = {lang: priors[lang].bits(' '.join(t[lang][1]).replace(' ', '')) / letters(' '.join(t[lang][1])) for lang in LANGUAGES}
    code = ['experiments/language_id.py', 'experiments/joint_recovery_v4.py', 'voynich/description_length.py',
            'voynich/joint_segments_v2.py', 'voynich/lexicon_repair.py', 'voynich/variable_units.py', 'voynich/decipher.py']
    write(OUT / 'freeze.json', dict(settings=four['settings'], prior_letters=n, held_out_bits_per_letter=entropy,
          prior_sha256={lang: hashlib.sha256(b''.join(p.tobytes() for p in priors[lang].probabilities)).hexdigest() for lang in LANGUAGES},
          code_sha256={p: digest(ROOT / p) for p in code}, protocol_sha256=digest(OUT / 'PROTOCOL.md'),
          sources_sha256=digest(ROOT / 'experiments/language-sources.json'), case_cap_seconds=CASE_CAP, at=time.time()))
    print('Frozen; commit before prepare', dict(prior_letters=n, entropy=entropy))


def verify():
    f = read(OUT / 'freeze.json')
    for p, h in list(f['code_sha256'].items()) + [('experiments/language-id/PROTOCOL.md', f['protocol_sha256'])]:
        if digest(ROOT / p) != h: raise ValueError('Frozen drift: ' + p)
    for p in list(f['code_sha256']) + ['experiments/language-id/PROTOCOL.md', 'experiments/language-id/freeze.json']:
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)).hexdigest() != digest(ROOT / p):
            raise ValueError('Freeze not committed: ' + p)
    return f


def italian_passage():
    released = set()
    for path in RELEASED:
        data = read(ROOT / path); rows = data['answers'] if isinstance(data, dict) else data
        for r in rows: released.update(i.split(':part')[0] for i in r['source_ids'])
    rows = [r for r in historical_rows()[0] if r['id'] not in released]
    blocks, _ = pack(rows, set())
    return blocks[0]


def prepare():
    f = verify()
    if (STATE / 'public.json').exists(): raise FileExistsError('Already prepared')
    t = texts(); vendor = load_vendor(); public, answers = [], []
    for lang in LANGUAGES:
        if lang == 'italian':
            block = italian_passage(); plain, source_ids = block['plaintext'], block['source_ids']
        else:
            plain, source_ids = t[lang][2], None
        prior_words = ' '.join(take(t[lang][0], f['prior_letters'])).split()
        seen = {tuple(prior_words[i:i + 20]) for i in range(len(prior_words) - 19)}
        words = plain.split()
        if any(tuple(words[i:i + 20]) in seen for i in range(len(words) - 19)): raise ValueError('Passage overlaps prior: ' + lang)
        dense = plain.replace(' ', '')
        seed = random.SystemRandom().randrange(2 ** 63); rng = random.Random(seed)
        alphabet = list(ALPHABET); rng.shuffle(alphabet); key = dict(zip(ALPHABET, alphabet)); inverse = {v: k for k, v in key.items()}
        random.seed(seed); trace = io.StringIO()
        tokens = vendor.encrypt_naibbe(''.join(key[c] for c in dense), vendor.naibbe_tables, vendor.placeholder_to_glyph,
                                       use_78=False, pre_plaintext_file=trace)
        if ''.join(inverse[c] for c in ''.join(trace.getvalue().split())) != dense: raise ValueError('Roundtrip failed')
        cipher = ' '.join(tokens); ident = hashlib.sha256((cipher + str(seed)).encode()).hexdigest()[:16]
        public.append(dict(id=ident, ciphertext=cipher))
        answers.append(dict(id=ident, language=lang, plaintext=plain, source_ids=source_ids, seed=seed, letters=len(dense)))
    private = STATE / 'evaluator-only'; private.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(STATE / 'public.json', sorted(public, key=lambda r: r['id'])); write(private / 'answers.json', answers)
    write(STATE / 'challenge.json', dict(public_sha256=digest(STATE / 'public.json'), answers_sha256=digest(private / 'answers.json'),
          freeze_sha256=digest(OUT / 'freeze.json'), head_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          letters=[a['letters'] for a in answers]))
    print('Prepared', len(public), 'ciphertexts; languages evaluator-only')


def decode(tokens, prior, s, cap):
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                      usage_floor=s['repair_usage_floor'], **em)
    units = role_units(repaired['segmentation'])
    ref = refine(units, prior, majority_key(units, repaired['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    return dict(recovered=ref['recovered'], refine_cap_hit=ref['cap_hit'], em_cap_hit=[first['cap_hit'], second['cap_hit'], repaired['cap_hit']],
                seconds=time.monotonic() - started)


def solve():
    f = verify(); t = texts(); priors = fit_priors(t, f['prior_letters'])
    for lang, p in priors.items():
        if hashlib.sha256(b''.join(x.tobytes() for x in p.probabilities)).hexdigest() != f['prior_sha256'][lang]: raise ValueError('Prior drift')
    cases = read(STATE / 'public.json'); started = time.time(); done = 0
    for case in cases:
        for lang in LANGUAGES:
            checkpoint = STATE / 'partial' / f"{case['id']}-{lang}.json"
            if checkpoint.exists(): done += 1; continue
            r = decode(case['ciphertext'].split(), priors[lang], f['settings'], f['case_cap_seconds'])
            recovered = r['recovered']
            r.update(bits_per_letter=priors[lang].bits(recovered) / len(recovered))
            write(checkpoint, dict(result=r, freeze_sha256=digest(OUT / 'freeze.json'), public_sha256=digest(STATE / 'public.json')))
            done += 1
            print(f"Language ID {done}/{len(cases) * len(LANGUAGES)}: prior {lang}, {r['seconds']:.0f}s ({time.time() - started:.0f}s total)", flush=True)
    write(STATE / 'predictions.json', dict(public_sha256=digest(STATE / 'public.json'), freeze_sha256=digest(OUT / 'freeze.json'),
          results={f"{c['id']}-{lang}": read(STATE / 'partial' / f"{c['id']}-{lang}.json")['result'] for c in cases for lang in LANGUAGES}))


def evaluate():
    f = verify(); c = read(STATE / 'challenge.json'); pred = read(STATE / 'predictions.json')
    if digest(STATE / 'evaluator-only/answers.json') != c['answers_sha256'] or pred['public_sha256'] != c['public_sha256']: raise ValueError('Drift')
    t = texts(); priors = fit_priors(t, f['prior_letters']); h = f['held_out_bits_per_letter']
    rows = []
    for a in read(STATE / 'evaluator-only/answers.json'):
        dense = a['plaintext'].replace(' ', '')
        decoded = {L: pred['results'][f"{a['id']}-{L}"] for L in LANGUAGES}
        excess = {L: decoded[L]['bits_per_letter'] - h[L] for L in LANGUAGES}
        raw = {L: decoded[L]['bits_per_letter'] for L in LANGUAGES}
        oracle = {L: priors[L].bits(dense) / len(dense) - h[L] for L in LANGUAGES}
        rank = sorted(LANGUAGES, key=excess.get)
        rows.append(dict(language=a['language'], excess=excess, raw_bits=raw, oracle_excess=oracle, predicted=rank[0],
                         raw_predicted=min(LANGUAGES, key=raw.get), oracle_predicted=min(LANGUAGES, key=oracle.get),
                         true_rank=rank.index(a['language']) + 1, margin=excess[rank[1]] - excess[rank[0]],
                         cap_hits={L: decoded[L]['refine_cap_hit'] or any(decoded[L]['em_cap_hit']) for L in LANGUAGES}))
    correct = sum(r['predicted'] == r['language'] for r in rows)
    margins = sorted(r['margin'] for r in rows if r['predicted'] == r['language'])
    out = dict(rows=rows, correct=correct, raw_correct=sum(r['raw_predicted'] == r['language'] for r in rows),
               oracle_correct=sum(r['oracle_predicted'] == r['language'] for r in rows), success=correct >= 4,
               median_margin_when_correct=margins[len(margins) // 2] if margins else None,
               held_out_bits_per_letter=h, prior_letters=f['prior_letters'], voynich_used=False)
    write(OUT / 'results.json', out)
    write(OUT / 'evaluated-records.json', dict(public=read(STATE / 'public.json'), answers=read(STATE / 'evaluator-only/answers.json'), predictions=pred))
    for r in rows:
        print(r['language'], '->', r['predicted'], 'rank', r['true_rank'], {L: round(v, 3) for L, v in r['excess'].items()})
    print('correct', correct, '/ 5; raw', out['raw_correct'], '; oracle', out['oracle_correct'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['freeze', 'verify', 'prepare', 'solve', 'evaluate'])
    print(globals()[p.parse_args().command]())
