"""Frozen rejection-and-transfer screen. See experiments/rejection-transfer-v2/PROTOCOL.md.

python -m experiments.rejection_transfer_v2 sources | freeze | verify | prepare | run | report
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import tarfile
import tempfile
import time
import urllib.request

from experiments import language_id
from experiments.historical_sources import verify as historical
from experiments.joint_development import load_vendor, majority_key, role_units
from voynich.corpora import conllu_sentences
from voynich.data import digest
from voynich.decipher import ALPHABET, normalize, edit_distance
from voynich.description_length import CharacterPrior
from voynich.joint_segments_v2 import joint_em, prune_and_rerun
from voynich.lexicon_repair import repair
from voynich.rejection import decide, stop_reason, transfer
from voynich.variable_units import refine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/rejection-transfer-v2'
STATE = ROOT / 'artifacts/rejection-transfer-v2'
LANGUAGES = language_id.LANGUAGES
ORDER = ('english', 'italian', 'latin', 'german', 'old_french') * 2
KINDS = ('positive', 'shuffle', 'copy')
MAX_WORKER_SECONDS = 12 * 3600
RELEASED = ('decipherment', 'segmentation', 'codebook-free', 'standard-decipherment',
            'joint-recovery', 'joint-recovery-v2', 'joint-recovery-v3', 'joint-recovery-v4',
            'joint-recovery-v5', 'joint-recovery-v6', 'language-id')


def read(path):
    return json.loads(Path(path).read_text())


def seal(path, value):
    """Atomic, no-clobber JSON. Even a crash cannot leave a reusable partial checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def grams(text):
    words = text.split()
    return {hashlib.blake2b(' '.join(words[i:i + 8]).encode(), digest_size=16).digest()
            for i in range(len(words) - 7)}


def overlaps(text, exact, shingles):
    return fingerprint(text) in exact or bool(grams(text) & shingles)


def exclusions():
    """Conservative: all original training files, not just the fitted prefixes."""
    exact, shingles, inputs = set(), set(), {}

    def add(text):
        text = normalize(text)
        if text:
            exact.add(fingerprint(text))
            shingles.update(grams(text))

    for source in read(ROOT / 'experiments/language-sources.json')['sources']:
        for record in source['files']:
            if record['path'].endswith('.conllu'):
                path = ROOT / record['path']
                if digest(path) != record['sha256']:
                    raise ValueError('Source drift: ' + str(path))
                inputs[record['path']] = record['sha256']
                for row in conllu_sentences(path):
                    add(' '.join(row['words']))
    for row in historical():
        for paragraph in row['paragraphs']:
            add(paragraph)
    for name in RELEASED:
        path = ROOT / f'artifacts/{name}/evaluator-only/answers.json'
        if not path.exists():
            continue
        # These are named, already graded recovery rounds; no other track's sealed answers.
        inputs[str(path.relative_to(ROOT))] = digest(path)
        rows = read(path)
        if isinstance(rows, dict):
            rows = rows['answers']
        for row in rows:
            if 'plaintext' in row:
                add(row['plaintext'])
    released = ROOT / 'artifacts/rejection-transfer/released-answers.json'
    inputs[str(released.relative_to(ROOT))] = digest(released)
    for row in read(released).values():
        for passage in row['passages']:
            add(passage['plaintext'])
    return exact, shingles, inputs


def used_source_ids(language):
    records = read(ROOT / 'artifacts/rejection-transfer/released-answers.json')
    return {i for row in records.values() if row['language'] == language
            for passage in row['passages'] for i in passage['source_ids']}


def pack_passages(rows, exact, shingles, count=4, excluded_ids=()):
    selected, sentences, ids, length = [], [], [], 0
    rejected = Counter()
    # Include previously retained candidates, so no near-duplicate crosses passage boundaries.
    exact, shingles = set(exact), set(shingles)
    for row in rows:
        if row['id'] in excluded_ids:
            rejected['previously_graded_source_id'] += 1
            continue
        text = normalize(' '.join(row['words']))
        n = len(text.replace(' ', ''))
        if not n:
            rejected['empty'] += 1
            continue
        if overlaps(text, exact, shingles):
            rejected['overlap'] += 1
            continue
        if length + n > 6000:
            rejected['too_long_for_current_passage'] += 1
            continue
        exact.add(fingerprint(text))
        shingles.update(grams(text))
        sentences.append(text)
        ids.append(row['id'])
        length += n
        if length >= 5200:
            selected.append(dict(plaintext=' '.join(sentences), source_ids=ids, letters=length))
            sentences, ids, length = [], [], 0
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} of {count} disjoint passages available; exclusions {dict(rejected)}')
    return selected, dict(rejected)


def sources():
    if (OUT / 'sources.json').exists():
        raise FileExistsError('Source audit already sealed')
    manifest = read(ROOT / 'experiments/language-sources.json')
    files = []
    for language, training in language_id.FILES.items():
        repository, name = training.split('/')
        source = next(s for s in manifest['sources'] if s['repository'] == repository)
        target_name = name.replace('-train.conllu', '-test.conllu')
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repository}/{source['revision']}/{target_name}"
        target = STATE / 'sources' / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            data = urllib.request.urlopen(url, timeout=60).read()
            with target.open('xb') as stream:
                stream.write(data)
        files.append(dict(language=language, repository=repository, revision=source['revision'],
                          license=source['license'], url=url, path=str(target.relative_to(ROOT)), sha256=digest(target)))
    pud = next(f for f in read(ROOT / 'experiments/modern-fresh-sources.json')['files'] if f['path'].endswith('.conllu'))
    if digest(ROOT / pud['path']) != pud['sha256']:
        raise ValueError('PUD drift')
    files.append(dict(pud, language='italian'))
    exact, shingles, inputs = exclusions()
    inventories = {}
    for record in files:
        passages, rejected = pack_passages(conllu_sentences(ROOT / record['path']), exact, shingles, excluded_ids=used_source_ids(record['language']))
        inventories[record['language']] = dict(passages=len(passages), letters=[p['letters'] for p in passages],
                                               excluded=rejected, reserved_sentences=sum(len(p['source_ids']) for p in passages))
    seal(OUT / 'sources.json', dict(files=files, exclusion_inputs=inputs, counts=inventories,
                                   sentence_policy='normalized exact plus 8-word overlap',
                                   historical_manifest_sha256=digest(ROOT / 'experiments/standard-decipherment/sources.json')))
    print(json.dumps(inventories, indent=2), flush=True)


def code_paths():
    extra = ['experiments/rejection_transfer_v2.py', 'tests/test_rejection_transfer.py', 'tests/test_rejection_transfer_v2.py',
             'experiments/language_id.py', 'experiments/joint_development.py',
             'experiments/joint_recovery.py', 'experiments/joint_recovery_v4.py',
             'experiments/historical_sources.py', 'experiments/segmentation.py',
             'experiments/word_segmentation_fresh.py', 'experiments/word_segmentation_v3_fresh.py',
             'experiments/word_segmentation_v3.py', 'experiments/word_segmentation_v2.py']
    return sorted(set(extra + [str(p.relative_to(ROOT)) for p in (ROOT / 'voynich').glob('*.py')]))


def freeze():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Already frozen')
    old = language_id.verify()
    # Round-four verification also pins every transitive decoder dependency/input it used.
    language_id.round_four()
    t = language_id.texts()
    priors = language_id.fit_priors(t, old['prior_letters'])
    prior_files = {}
    for language, prior in priors.items():
        value = hashlib.sha256(b''.join(x.tobytes() for x in prior.probabilities)).hexdigest()
        if value != old['prior_sha256'][language]:
            raise ValueError('Prior changed: ' + language)
        path = STATE / 'priors' / (language + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        prior.save(path)
        prior_files[str(path.relative_to(ROOT))] = digest(path)
    old_results = read(ROOT / 'experiments/language-id/results.json')
    calibration = dict(source='released language-id pilot, now development evidence',
                       positives=[dict(language=r['language'], excess=r['excess'][r['language']], margin=r['margin'])
                                  for r in old_results['rows']], excess_ceiling=.50, margin=.25, coverage=.95,
                       null_calibration=False, fresh_test_used=False)
    seal(OUT / 'calibration.json', calibration)
    paths = code_paths() + ['experiments/rejection-transfer-v2/PROTOCOL.md', 'experiments/rejection-transfer-v2/sources.json',
                            'experiments/rejection-transfer-v2/calibration.json', 'experiments/language-id/freeze.json',
                            'experiments/language-id/results.json', 'experiments/decipherment-sources.json']
    source = read(OUT / 'sources.json')
    external = dict(prior_files, **source['exclusion_inputs'])
    external.update({f['path']: f['sha256'] for f in source['files']})
    external.update({f['path']: f['sha256'] for f in read(ROOT / 'experiments/decipherment-sources.json')['files']})
    seal(OUT / 'freeze.json', dict(code_and_records={p: digest(ROOT / p) for p in paths}, external=external,
                                  settings=dict(old['settings'], refine_cap=1200), entropy=old['held_out_bits_per_letter'],
                                  prior_files=prior_files, fit_cap=old['case_cap_seconds'], workers=5,
                                  order=ORDER, max_worker_seconds=MAX_WORKER_SECONDS, at=time.time()))
    print('Resource-repair round frozen: 1,200-second refinement cap, two Numba threads per worker. Commit before prepare.', flush=True)


def verify(require_commit=True):
    f = read(OUT / 'freeze.json')
    for path, expected in {**f['code_and_records'], **f['external']}.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen drift: ' + path)
    if require_commit:
        for path in [*f['code_and_records'], 'experiments/rejection-transfer-v2/freeze.json']:
            committed = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != digest(ROOT / path):
                raise ValueError('Freeze not committed: ' + path)
    return f


def mutate(token, alphabet, rng):
    position = rng.randrange(len(token))
    action = rng.randrange(3)
    if action == 0:
        return token[:position] + rng.choice(alphabet) + token[position:]
    if action == 1 and len(token) > 1:
        return token[:position] + token[position + 1:]
    choices = [g for g in alphabet if g != token[position]]
    return token[:position] + rng.choice(choices or alphabet) + token[position + 1:]


def copy_text(inventory, length, seed):
    rng = random.Random(seed)
    alphabet = sorted(set(''.join(inventory)))
    out = [rng.choice(inventory) for _ in range(min(32, length))]
    while len(out) < length:
        draw = rng.random()
        if draw < .8:
            token = rng.choice(out[-50:])
            out.append(token if draw < .6 else mutate(token, alphabet, rng))
        else:
            out.append(rng.choice(inventory))
    return out


def encrypt(vendor, plaintext, key, seed):
    random.seed(seed)
    trace = io.StringIO()
    tokens = vendor.encrypt_naibbe(''.join(key[c] for c in plaintext.replace(' ', '')),
                                  vendor.naibbe_tables, vendor.placeholder_to_glyph,
                                  use_78=False, pre_plaintext_file=trace)
    inverse = {v: k for k, v in key.items()}
    if ''.join(inverse[c] for c in ''.join(trace.getvalue().split())) != plaintext.replace(' ', ''):
        raise ValueError('Encryption round trip failed')
    return tokens


def prepare():
    f = verify()
    if (STATE / 'challenge.json').exists() or (STATE / 'public').exists():
        raise FileExistsError('Already prepared, or incomplete preparation needs investigation')
    exact, shingles, _ = exclusions()
    source = read(OUT / 'sources.json')
    pools = {r['language']: pack_passages(conllu_sentences(ROOT / r['path']), exact, shingles, excluded_ids=used_source_ids(r['language']))[0] for r in source['files']}
    vendor = load_vendor()
    rng = random.SystemRandom()
    answers, schedule, inputs = {}, [], {}
    occurrences = Counter()
    private = STATE / 'evaluator-only'
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    for block_number, language in enumerate(f['order']):
        offset = occurrences[language] * 2
        occurrences[language] += 1
        passages = pools[language][offset:offset + 2]
        key_seed = rng.randrange(2 ** 63)
        letters = list(ALPHABET)
        random.Random(key_seed).shuffle(letters)
        key = dict(zip(ALPHABET, letters))
        seeds = [rng.randrange(2 ** 63) for _ in range(6)]
        positive = [encrypt(vendor, p['plaintext'], key, s) for p, s in zip(passages, seeds[:2])]
        shuffled = [list(t) for t in positive]
        for tokens, seed in zip(shuffled, seeds[2:4]):
            random.Random(seed).shuffle(tokens)
        copied = [copy_text(positive[0], len(t), seed) for t, seed in zip(positive, seeds[4:])]
        ids = []
        for kind, pair in zip(KINDS, [positive, shuffled, copied]):
            ident = fingerprint(str(rng.randrange(2 ** 128)))[:20]
            ids.append(ident)
            for role, tokens in zip(('fit', 'transfer'), pair):
                path = STATE / 'public' / f'{ident}-{role}.json'
                seal(path, dict(id=ident, ciphertext=' '.join(tokens)))
                inputs[str(path.relative_to(ROOT))] = digest(path)
            answers[ident] = dict(language=language, kind=kind, block=block_number, passages=passages,
                                 key_seed=key_seed, seeds=seeds, key=key)
        schedule.append(dict(block=block_number, inputs=ids))
    seal(private / 'answers.json', answers)
    seal(STATE / 'challenge.json', dict(schedule=schedule, inputs=inputs, answers_sha256=digest(private / 'answers.json'),
                                      freeze_sha256=digest(OUT / 'freeze.json'),
                                      freeze_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                      at=time.time()))
    print('Prepared 10 sealed blocks, 30 paired inputs. No plaintext exposed.', flush=True)


def checked_public(ident, role, challenge):
    path = STATE / 'public' / f'{ident}-{role}.json'
    if digest(path) != challenge['inputs'][str(path.relative_to(ROOT))]:
        raise ValueError('Public challenge drift')
    value = read(path)
    if value['id'] != ident:
        raise ValueError('Public ID drift')
    return value['ciphertext'].split()


def provenance(ident, language, challenge):
    return dict(id=ident, language=language, freeze_sha256=digest(OUT / 'freeze.json'),
                challenge_sha256=digest(STATE / 'challenge.json'), inputs=challenge['inputs'])


def checked_checkpoint(path, expected):
    value = read(path)
    if value['provenance'] != expected:
        raise ValueError('Checkpoint provenance drift: ' + str(path))
    return value['result']


def fit_key(tokens, prior, s, cap):
    """The old language-ID decode verbatim in stages; additionally expose its final key."""
    started = time.monotonic()
    em = dict(minimum=s['minimum'], restarts=s['joint_restarts'], iterations=s['joint_iterations'], seed=s['seed'], cap=cap * .25)
    first = joint_em(tokens, prior, **em)
    second = prune_and_rerun(tokens, prior, first, minimum_usage=s['prune_usage'], **em)
    repaired = repair(tokens, prior, second, theta=s['repair_theta'], complement_minimum=s['repair_minimum'], passes=s['repair_passes'],
                      usage_floor=s['repair_usage_floor'], **em)
    units = role_units(repaired['segmentation'])
    ref = refine(units, prior, majority_key(units, repaired['recovered']), (), seed=1, kicks=s['refine_kicks'], kick_size=s['refine_kick_size'],
                 cap=min(s['refine_cap'], max(30., cap - (time.monotonic() - started))))
    elapsed = time.monotonic() - started
    hits = dict(first=first['cap_hit'], second=second['cap_hit'], repair=repaired['cap_hit'], refine=ref['cap_hit'], case=elapsed > cap)
    return dict(recovered=ref['recovered'], mapping=ref['mapping'], cap_hits=hits, cap_hit=any(hits.values()),
                bits_per_letter=prior.bits(ref['recovered']) / len(ref['recovered']), seconds=elapsed)


def worker(stage, ident, language):
    # Supervisor verifies all frozen code, inputs and commit before dispatching workers.
    f, challenge = read(OUT / 'freeze.json'), read(STATE / 'challenge.json')
    expected = provenance(ident, language, challenge)
    path = STATE / stage / f'{ident}-{language}.json'
    if path.exists():
        checked_checkpoint(path, expected)
        return str(path)
    prior_path = STATE / 'priors' / (language + '.npz')
    if digest(prior_path) != f['prior_files'][str(prior_path.relative_to(ROOT))]:
        raise ValueError('Worker prior drift')
    prior = CharacterPrior.load(prior_path)
    if stage == 'fit':
        # Deliberately never load transfer ciphertext in this branch.
        result = fit_key(checked_public(ident, 'fit', challenge), prior, f['settings'], f['fit_cap'])
    else:
        key_seal = read(STATE / 'sealed-keys' / (ident + '.json'))
        key_path = STATE / 'fit' / f'{ident}-{language}.json'
        if digest(key_path) != key_seal['key_files'][str(key_path.relative_to(ROOT))]:
            raise ValueError('Learned key changed after sealing')
        fitted = checked_checkpoint(key_path, expected)
        result = transfer(checked_public(ident, 'transfer', challenge), fitted['mapping'], prior)
        result['key_sha256'] = fingerprint(json.dumps(fitted['mapping'], sort_keys=True))
    seal(path, dict(provenance=expected, result=result, at=time.time()))
    return str(path)


def evaluate_input(ident, challenge):
    pred_path = STATE / 'predictions' / (ident + '.json')
    predictions = read(pred_path)
    if predictions['challenge_sha256'] != digest(STATE / 'challenge.json'):
        raise ValueError('Prediction challenge drift')
    for path, sha in predictions['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Prediction drift')
    answers_path = STATE / 'evaluator-only/answers.json'
    if digest(answers_path) != challenge['answers_sha256']:
        raise ValueError('Answer drift')
    gold = read(answers_path)[ident]
    rows = {}
    f = read(OUT / 'freeze.json')
    for language in LANGUAGES:
        expected = provenance(ident, language, challenge)
        fitted = checked_checkpoint(STATE / 'fit' / f'{ident}-{language}.json', expected)
        held = checked_checkpoint(STATE / 'transfer' / f'{ident}-{language}.json', expected)
        rows[language] = dict(fit_excess=fitted['bits_per_letter'] - f['entropy'][language],
                              transfer_excess=None if held['bits_per_letter'] is None else held['bits_per_letter'] - f['entropy'][language],
                              coverage=held['token_coverage'], glyph_coverage=held['glyph_coverage'],
                              cap_hit=fitted['cap_hit'], cap_hits=fitted['cap_hits'], seconds=fitted['seconds'])
        if gold['kind'] == 'positive':
            for label, result, passage in zip(('fit', 'transfer'), (fitted, held), gold['passages']):
                dense = passage['plaintext'].replace(' ', '')
                rows[language][label + '_cer'] = edit_distance(result['recovered'], dense) / len(dense)
    outcomes = [dict(id=ident, block=gold['block'], kind=gold['kind'], language=gold['language'],
                     decision=decide(rows), scores=rows, predictions_sha256=digest(pred_path))]
    if gold['kind'] == 'positive':
        outcomes.append(dict(id=ident, block=gold['block'], kind='absent', language=gold['language'],
                             decision=decide(rows, [l for l in LANGUAGES if l != gold['language']]),
                             predictions_sha256=digest(pred_path)))
    return outcomes


def run():
    f = verify()
    if (OUT / 'results.json').exists():
        raise FileExistsError('This screen has already ended')
    challenge = read(STATE / 'challenge.json')
    if challenge['freeze_sha256'] != digest(OUT / 'freeze.json'):
        raise ValueError('Challenge freeze drift')
    outcomes, reason = [], None
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '2' if name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS') else '1'
    with ProcessPoolExecutor(max_workers=f['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for block in challenge['schedule']:
            for ident in block['inputs']:
                scored = STATE / 'graded' / (ident + '.json')
                if scored.exists():
                    old = read(scored)
                    # Verify prediction files before accepting a previously graded checkpoint.
                    pred = read(STATE / 'predictions' / (ident + '.json'))
                    if old['predictions_sha256'] != digest(STATE / 'predictions' / (ident + '.json')) or \
                            any(digest(ROOT / p) != h for p, h in pred['files'].items()):
                        raise ValueError('Graded checkpoint drift')
                    outcomes.extend(old['outcomes'])
                    reason = stop_reason(outcomes)
                    if reason:
                        break
                    continue
                used = sum(read(p)['result']['seconds'] for p in (STATE / 'fit').glob('*.json'))
                if used + len(LANGUAGES) * f['fit_cap'] > f['max_worker_seconds']:
                    reason = 'inconclusive_worker_budget'
                    break
                print(f"Block {block['block'] + 1}/10: fitting five priors for input {ident}", flush=True)
                futures = [pool.submit(worker, 'fit', ident, l) for l in LANGUAGES]
                paths = [Path(job.result()) for job in futures]
                key_path = STATE / 'sealed-keys' / (ident + '.json')
                key_record = dict(key_files={str(p.relative_to(ROOT)): digest(p) for p in paths},
                                  freeze_sha256=digest(OUT / 'freeze.json'))
                if key_path.exists():
                    if read(key_path) != key_record:
                        raise ValueError('Sealed keys drift')
                else:
                    seal(key_path, key_record)
                print(f"Block {block['block'] + 1}/10: keys sealed; transferring without refitting", flush=True)
                futures = [pool.submit(worker, 'transfer', ident, l) for l in LANGUAGES]
                paths += [Path(job.result()) for job in futures]
                pred_path = STATE / 'predictions' / (ident + '.json')
                pred = dict(files={str(p.relative_to(ROOT)): digest(p) for p in paths},
                            challenge_sha256=digest(STATE / 'challenge.json'))
                if pred_path.exists():
                    if read(pred_path) != pred:
                        raise ValueError('Prediction seal drift')
                else:
                    seal(pred_path, pred)
                graded = evaluate_input(ident, challenge)
                seal(scored, dict(outcomes=graded, predictions_sha256=digest(pred_path)))
                outcomes.extend(graded)
                reason = stop_reason(outcomes)
                for row in graded:
                    print(json.dumps({k: row[k] for k in ('block', 'kind', 'language', 'decision')}), flush=True)
                if reason:
                    break
            if reason:
                break
    summary = {}
    for kind in (*KINDS, 'absent'):
        rows = [r for r in outcomes if r['kind'] == kind]
        summary[kind] = dict(evaluated=len(rows), accepted=sum(r['decision']['accepted'] is not None for r in rows),
                             correct=sum(not r['decision']['inconclusive'] and r['decision']['accepted'] == (r['language'] if kind == 'positive' else None) for r in rows),
                             inconclusive=sum(r['decision']['inconclusive'] for r in rows))
    seal(OUT / 'results.json', dict(status='screen_passed' if reason is None else
                                   ('inconclusive' if reason.startswith('inconclusive') else 'screen_failed'),
                                   stop_reason=reason, summary=summary, outcomes=outcomes,
                                   independent_positive_keys=summary['positive']['evaluated'],
                                   planned_blocks=10, fit_worker_seconds=sum(read(p)['result']['seconds'] for p in (STATE / 'fit').glob('*.json')),
                                   freeze_sha256=digest(OUT / 'freeze.json'), challenge_sha256=digest(STATE / 'challenge.json'),
                                   voynich_used=False, manuscript_run_licensed=False, at=time.time()))
    print(json.dumps(read(OUT / 'results.json')['summary'], indent=2), flush=True)


def report():
    verify()
    result = read(OUT / 'results.json')
    if result['freeze_sha256'] != digest(OUT / 'freeze.json') or result['challenge_sha256'] != digest(STATE / 'challenge.json'):
        raise ValueError('Report provenance drift')
    lines = ['# Rejection and frozen-key transfer: ' + result['status'].replace('_', ' '), '',
             '## What was done', '', '- Tested the frozen five-language decoder on fresh fit/transfer pairs.',
             '- Sealed learned keys before loading transfer ciphertext; no transfer key refit.',
             '- Used the declared early-stop rule. No Voynich text was scored.', '', '## Why', '',
             'A decoder that always produces text needs a tested rejection rule. A mapping must also work on text it was not fitted to.', '',
             '## Results', '', '| Input | Evaluated | Accepted | Correct decision |', '|---|---:|---:|---:|']
    for kind, counts in result['summary'].items():
        lines.append(f"| {kind} | {counts['evaluated']} | {counts['accepted']} | {counts['correct']} |")
    lines += ['', f"**Stop:** `{result['stop_reason']}`. Planned: 10 independent passage/key blocks. "
              f"Evaluated positives: {result['independent_positive_keys']}. Fit-worker time: {result['fit_worker_seconds'] / 3600:.2f} hours.", '',
              '| Block | Input | Source language | Fit winner / excess | Transfer winner / excess | Accepted | Reasons |',
              '|---|---|---|---|---|---|---|']
    for row in result['outcomes']:
        d = row['decision']
        def value(r):
            return 'unscorable' if r['winner'] is None else f"{r['winner']} / {r['excess']:.3f}"
        lines.append(f"| {row['block'] + 1} | {row['kind']} | {row['language']} | {value(d['fit'])} | {value(d['transfer'])} | "
                     f"{d['accepted'] or 'none'} | {', '.join(d['reasons']) or 'passed'} |")
    lines += ['', '## Positive recovery diagnostics', '',
              '| Block | True language | Fit CER | Transfer CER | Transfer token coverage |', '|---|---|---:|---:|---:|']
    for row in result['outcomes']:
        if row['kind'] == 'positive':
            s = row['scores'][row['language']]
            lines.append(f"| {row['block'] + 1} | {row['language']} | {s['fit_cer']:.2%} | {s['transfer_cer']:.2%} | {s['coverage']:.2%} |")
    lines += ['', '## Limits and decision', '',
              'This is a bounded screen with thresholds chosen from the released five-case language-ID pilot. '
              'It is not a fresh estimate of a 90% sensitivity / 5% false-positive operating point. '
              'Paired negatives share their positive source and are not independent replicates. '
              'Early-stop proportions are descriptive; unrun cases are not counted as rejections.', '',
              'The old language-ID decoder with a 1,200-second refinement cap and two Numba threads, and its default Naibbe spacing, were tested. '
              'The newer long-passage Italian decoder, heavier pairing, other cipher families, '
              'other languages, and the published Timm generator were not tested.', '',
              '**No manuscript run is licensed.** A failed screen parks this version; an inconclusive '
              'screen needs a declared resource/protocol repair; a pass would require a larger independent control study.', '',
              '## Records and reproduction', '',
              '[Protocol](PROTOCOL.md) · [Freeze](freeze.json) · [Calibration](calibration.json) · '
              '[Sources](sources.json) · [Results](results.json) · [Released records](evaluated-records.tar.gz)', '',
              '```sh', '.venv/bin/python -m experiments.rejection_transfer_v2 verify',
              '.venv/bin/python -m unittest tests.test_rejection_transfer -v', '```', '',
              'Driver subcommands: `sources`, `freeze`, commit, `prepare`, `run`, `report`. '
              'Creation and grading refuse overwrites. `run` resumes verified per-input checkpoints. '
              'Archive contains only graded answers; unrun answers remain evaluator-only.', '']
    target = OUT / 'REPORT.md'
    with target.open('x') as stream:
        stream.write('\n'.join(lines))
    ids = sorted({r['id'] for r in result['outcomes']})
    answers = read(STATE / 'evaluator-only/answers.json')
    released = STATE / 'released-answers.json'
    seal(released, {ident: answers[ident] for ident in ids})
    archive = OUT / 'evaluated-records.tar.gz'
    with archive.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as tar:
        tar.add(released, arcname='answers.json')
        for ident in ids:
            for role in ('fit', 'transfer'):
                tar.add(STATE / 'public' / f'{ident}-{role}.json', arcname=f'public/{ident}-{role}.json')
            for directory in ('fit', 'transfer'):
                for language in LANGUAGES:
                    name = f'{ident}-{language}.json'
                    tar.add(STATE / directory / name, arcname=f'{directory}/{name}')
            for directory in ('sealed-keys', 'predictions', 'graded'):
                tar.add(STATE / directory / (ident + '.json'), arcname=f'{directory}/{ident}.json')
    print(target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['sources', 'freeze', 'verify', 'prepare', 'run', 'report'])
    command = parser.parse_args().command
    if command == 'verify':
        verify()
        print('Frozen code, inputs and commit verified.')
    else:
        globals()[command]()
