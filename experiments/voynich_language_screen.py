"""Pre-registered language screen on the non-reserved Voynich pages: one run, no tuning.

python -m experiments.voynich_language_screen prepare | run | evaluate

The confirmed screen (decoder A, eight v2 priors, `decide_transfer`) is applied to four blocks of
manuscript text from the train and validation pages of the fixed folio split. Reserved (test) pages
are never read. Each block has a fit passage and a transfer passage from later folios, plus the
same two negatives as every confirmation: a token shuffle and a frequency copy. There is no answer
file, because nothing is known; the pre-registered interpretation is in PROTOCOL.md.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import subprocess
import time

from experiments import key_recovery_confirmation_v2 as v2
from experiments import key_recovery_development as development
from experiments.rejection_transfer_v2 import read, seal
from voynich.data import digest
from voynich.description_length import CharacterPrior
from voynich.joint_segments import candidate_pieces
from voynich.rejection import transfer
from voynich.rejection_development import decide_transfer, frequency_copy
from voynich.rejection_v3 import group_scores, rescore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/voynich-language-screen'
STATE = ROOT / 'artifacts/voynich-language-screen'
DATA = ROOT / 'artifacts/data/zl/documents.json'          # Zandbergen-Landini transcription, Eva
GLYPHS = set('acdefghiklmnopqrstxy')                       # every character the Naibbe encoder can emit
SEPARATORS = re.compile(r'[.,\n\x1c\x1d\s]+')               # word space, uncertain space, line, drawing marks
KINDS = ('voynich', 'shuffle', 'frequency_copy')
MIN_COVERAGE = .95
# Fixed before the run. Slices are token offsets into the group's clean tokens in manuscript order.
BLOCKS = (
    dict(name='herbal_A', fit=[('herbal', 'A', 0, 3400)], transfer=[('herbal', 'A', 3400, 6800)]),
    dict(name='stars_B', fit=[('stars', 'B', 0, 3400)], transfer=[('stars', 'B', 3400, 6800)]),
    dict(name='balneological_B', fit=[('balneological', 'B', 0, 2600)], transfer=[('balneological', 'B', 2600, 5200)]),
    dict(name='mixed_B', fit=[('herbal', 'B', 0, None)],
         transfer=[('text', 'B', 0, None), ('cosmological', 'B', 0, None), ('stars', 'B', 6800, None)]),
)


def clean_tokens(text):
    """Split on every separator; drop tokens with a glyph the cipher cannot produce (mostly '?')."""
    return [t for t in SEPARATORS.split(text) if t and set(t) <= GLYPHS]


def pages():
    return [p for p in read(DATA) if p['split'] != 'test']


def group_tokens(documents, section, currier):
    out, used = [], []
    for p in documents:
        if p['section'] == section and p['currier'] == currier:
            out.extend(clean_tokens(p['text']))
            used.append(p['page'])
    return out, used


def passage(documents, spec):
    tokens, sources = [], []
    for section, currier, start, stop in spec:
        group, used = group_tokens(documents, section, currier)
        part = group[start:stop]
        if len(part) < (stop or len(group)) - start:
            raise ValueError(f'Group {section}/{currier} has only {len(group)} tokens')
        tokens.extend(part)
        sources.append(dict(section=section, currier=currier, start=start, stop=stop, tokens=len(part), pages=used))
    return tokens, sources


def readability(fit, held):
    """Share of transfer tokens a key built from the fit passage's frequent pieces could read."""
    pieces = candidate_pieces(fit, 6)
    ok = sum(1 for t in held if t in pieces or any(t[:i] in pieces and t[i:] in pieces for i in range(1, len(t))))
    return ok / len(held)


def prepare():
    if (STATE / 'challenge.json').exists():
        raise FileExistsError('Already prepared')
    documents = pages()
    if any(p['split'] == 'test' for p in documents):
        raise ValueError('Reserved page in the working set')
    raw = sum(len([t for t in SEPARATORS.split(p['text']) if t]) for p in documents)
    kept = sum(len(clean_tokens(p['text'])) for p in documents)
    rng = random.SystemRandom()
    blocks, inputs = [], {}
    for spec in BLOCKS:
        fit, fit_sources = passage(documents, spec['fit'])
        held, held_sources = passage(documents, spec['transfer'])
        if set(t for s in fit_sources for t in s['pages']) & set(p for s in held_sources for p in s['pages']) and \
                not all(s['stop'] is not None or s['start'] for s in spec['fit'] + spec['transfer']):
            raise ValueError('Fit and transfer share pages without an offset: ' + spec['name'])
        seeds = [rng.randrange(2 ** 63) for _ in range(4)]
        shuffled = [list(t) for t in (fit, held)]
        for tokens, seed in zip(shuffled, seeds[:2]):
            random.Random(seed).shuffle(tokens)
        frequency = [frequency_copy(t, seed) for t, seed in zip((fit, held), seeds[2:])]
        ids = {}
        for kind, pair in zip(KINDS, ((fit, held), shuffled, frequency)):
            ident = hashlib.sha256(str(rng.randrange(2 ** 128)).encode()).hexdigest()[:20]
            ids[kind] = ident
            for role, tokens in zip(('fit', 'transfer'), pair):
                path = STATE / 'public' / f'{ident}-{role}.json'
                seal(path, dict(id=ident, ciphertext=' '.join(tokens)))
                inputs[str(path.relative_to(ROOT))] = digest(path)
        blocks.append(dict(name=spec['name'], ids=ids, seeds=seeds, fit=fit_sources, transfer=held_sources,
                           fit_tokens=len(fit), transfer_tokens=len(held), readability=readability(fit, held),
                           fit_types=len(set(fit)), fit_hapax_share=sum(1 for t in set(fit) if fit.count(t) == 1) / len(set(fit))))
    seal(STATE / 'challenge.json', dict(
        blocks=blocks, inputs=inputs, kinds=KINDS, data_sha256=digest(DATA), pages=len(documents),
        raw_tokens=raw, kept_tokens=kept, dropped_share=1 - kept / raw,
        prior_freeze_sha256=digest(v2.OUT / 'freeze.json'),
        protocol_sha256=digest(OUT / 'PROTOCOL.md'),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), at=time.time()))
    for b in blocks:
        print(f"{b['name']:16s} fit {b['fit_tokens']} transfer {b['transfer_tokens']} readability {b['readability']:.3f}", flush=True)


def worker(job):
    ident, model = job
    path = STATE / 'fit' / f'{ident}-{model}.json'
    if path.exists():
        return str(path)
    challenge = read(STATE / 'challenge.json')
    public = STATE / 'public' / f'{ident}-fit.json'
    if digest(public) != challenge['inputs'][str(public.relative_to(ROOT))]:
        raise ValueError('Input drift')
    prior = CharacterPrior.load(v2.prior_path(model))
    fitted = development.fit_both(read(public)['ciphertext'].split(), prior, read(v2.PRIOR_FREEZE))
    held = read(STATE / 'public' / f'{ident}-transfer.json')['ciphertext'].split()
    for arm in ('B', 'A'):
        fitted[arm]['transfer'] = transfer(held, fitted[arm]['mapping'], prior)
    seal(path, dict(id=ident, model=model, result=fitted))
    return str(path)


def run(workers):
    for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '2'
    models = list(read(v2.OUT / 'freeze.json')['labels'].values())
    jobs = [(i, m) for b in read(STATE / 'challenge.json')['blocks'] for i in b['ids'].values() for m in models]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for done, _ in enumerate(pool.map(worker, jobs), 1):
            if done % 8 == 0:
                print(f'{done}/{len(jobs)} fits', flush=True)
    print('Fitted', len(jobs), 'input-model pairs.', flush=True)


def outcome(decision, coverage_gate=MIN_COVERAGE):
    """Pre-registered classes: accepted, inconclusive, unreadable, rejected."""
    if decision['inconclusive']:
        return 'inconclusive'
    if decision['accepted'] is not None:
        return 'accepted'
    if 'coverage' in decision['reasons']:
        return 'unreadable'
    return 'rejected'


def reference_bands():
    """Score bands of the released confirmations, for the report only."""
    bands = {}
    for name, path in (('respacing17_confirmation', ROOT / 'experiments/key-recovery-confirmation-v2/results.json'),
                       ('respacing9_development', ROOT / 'experiments/respacing9-development/results.json')):
        rows = read(path)['rows']
        scores = lambda r: r['A']['scores'] if 'A' in r else None
        positives = [r for r in rows if r['kind'] == 'positive']
        negatives = [r for r in rows if r['kind'] != 'positive']
        if name == 'respacing9_development':   # scores are not stored per row there; skip the bands
            continue
        pf = sorted(scores(r)[r['language']]['fit_excess'] for r in positives)
        pt = sorted(scores(r)[r['language']]['transfer_excess'] for r in positives)
        nf = sorted(min(s['fit_excess'] for s in scores(r).values()) for r in negatives)
        nt = sorted(min(s['transfer_excess'] for s in scores(r).values() if s['transfer_excess'] is not None) for r in negatives)
        q = lambda a: dict(min=a[0], median=a[len(a) // 2], max=a[-1])
        bands[name] = dict(positive_true_model=dict(fit_excess=q(pf), transfer_excess=q(pt)),
                           negative_best_model=dict(fit_excess=q(nf), transfer_excess=q(nt)))
    return bands


def evaluate():
    freeze = read(v2.OUT / 'freeze.json')
    labels, entropy = freeze['labels'], freeze['entropy']
    priors = {m: CharacterPrior.load(v2.prior_path(m)) for m in labels.values()}
    challenge = read(STATE / 'challenge.json')
    rows = []
    for block in challenge['blocks']:
        for kind, ident in block['ids'].items():
            fits = {m: read(STATE / 'fit' / f'{ident}-{m}.json')['result'] for m in labels.values()}
            row = dict(block=block['name'], kind=kind, id=ident)
            for arm in ('A', 'B'):
                table = {}
                for lang, model in labels.items():
                    r = fits[model][arm]
                    honest = rescore(r['transfer'], priors[model])
                    table[lang] = dict(fit_excess=r['bits_per_letter'] - entropy[model],
                                       transfer_excess=None if r['transfer']['bits_per_letter'] is None
                                       else r['transfer']['bits_per_letter'] - entropy[model],
                                       transfer_excess_one_code=None if honest['bits_per_letter'] is None
                                       else honest['bits_per_letter'] - entropy[model],
                                       coverage=r['transfer']['token_coverage'], cap_hit=r['cap_hit'],
                                       fit_cap_hits=r.get('cap_hits'), recovered_letters=r['transfer']['recovered_letters'])
                primary_scores = {l: dict(fit_excess=s['fit_excess'], transfer_excess=s['transfer_excess'],
                                          coverage=s['coverage'], cap_hit=s['cap_hit']) for l, s in table.items()}
                candidate_scores = group_scores({l: dict(fit_excess=s['fit_excess'], transfer_excess=s['transfer_excess_one_code'],
                                                         coverage=s['coverage'], cap_hit=s['cap_hit']) for l, s in table.items()})
                primary, candidate = decide_transfer(primary_scores), decide_transfer(candidate_scores)
                row[arm] = dict(scores=table, primary=primary, primary_outcome=outcome(primary),
                                candidate=candidate, candidate_outcome=outcome(candidate),
                                best_fit=min(table, key=lambda l: table[l]['fit_excess']),
                                best_transfer=min((l for l in table if table[l]['transfer_excess'] is not None),
                                                  key=lambda l: table[l]['transfer_excess'], default=None))
            rows.append(row)
    summary = {}
    for kind in KINDS:
        summary[kind] = {rule: dict((o, sum(1 for r in rows if r['kind'] == kind and r['A'][f'{rule}_outcome'] == o))
                                    for o in ('accepted', 'rejected', 'unreadable', 'inconclusive'))
                         for rule in ('primary', 'candidate')}
    seal(OUT / 'results.json', dict(rows=rows, summary=summary, reference=reference_bands(), challenge=challenge,
                                    reserved_pages_used=False, decoder='A', at=time.time()))
    for r in rows:
        a = r['A']
        best = a['best_fit']
        s = a['scores'][best]
        print(f"{r['block']:16s} {r['kind']:15s} primary {a['primary_outcome']:12s} candidate {a['candidate_outcome']:12s} "
              f"best fit {best:10s} fit {s['fit_excess']:+.3f} transfer {s['transfer_excess'] if s['transfer_excess'] is None else round(s['transfer_excess'], 3)} "
              f"coverage {s['coverage']:.3f} reasons {a['primary']['reasons']}")
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    run(args.workers) if args.command == 'run' else globals()[args.command]()
