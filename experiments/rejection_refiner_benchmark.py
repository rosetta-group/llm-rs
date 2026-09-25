"""Fixed-work, isolated-process benchmark of released-data pair-swap search."""
import argparse
import gc
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

import numpy as np

from experiments.rejection_transfer_v2 import seal
from voynich.context_reparse import reparse
from voynich.description_length import CharacterPrior
from voynich.variable_units import key_arrays
from voynich.variable_units_bounded import Scorer, pair_batches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/rejection-followups'
STATE = ROOT / 'artifacts/rejection-followups'


def read(path):
    return json.loads(Path(path).read_text())


def fixtures():
    plan = read(OUT / 'BENCHMARK_PLAN.json')
    alphabet = CharacterPrior.load(ROOT / 'artifacts/rejection-transfer-v2/priors/english.npz').alphabet
    result = []
    for spec in plan['synthetic']:
        rng = np.random.default_rng(spec['seed'])
        inventory = ['u' + str(i) for i in range(spec['units'])]
        units = inventory + list(rng.choice(inventory, spec['positions'] - spec['units']))
        mapping = {u: str(rng.choice(list(alphabet))) for u in inventory}
        result.append(dict(name=spec['name'], units=units, mapping=mapping, prior='english'))
    with tarfile.open(ROOT / 'experiments/rejection-transfer-v2/evaluated-records.tar.gz') as archive:
        for spec in plan['released']:
            fitted = json.load(archive.extractfile(f"fit/{spec['id']}-{spec['prior']}.json"))['result']
            tokens = json.load(archive.extractfile(f"public/{spec['id']}-fit.json"))['ciphertext'].split()
            prior = CharacterPrior.load(ROOT / f"artifacts/rejection-transfer-v2/priors/{spec['prior']}.npz")
            parsed = reparse(tokens, fitted['mapping'], prior)
            units = [r + ':' + p for parts in parsed['segmentation']
                     for r, p in zip(('u',) if len(parts) == 1 else ('p', 's'), parts)]
            result.append(dict(name=spec['name'], units=units, mapping=fitted['mapping'], prior=spec['prior'],
                               source_id=spec['id'], parse='fixed-key context reparse; benchmark fixture only'))
    for fixture in result:
        seal(STATE / 'benchmark-inputs' / (fixture['name'] + '.json'), fixture)
    return result


def worker(name, backend):
    plan = read(OUT / 'BENCHMARK_PLAN.json')
    fixture = read(STATE / 'benchmark-inputs' / (name + '.json'))
    prior = CharacterPrior.load(ROOT / f"artifacts/rejection-transfer-v2/priors/{fixture['prior']}.npz")
    scorer = Scorer(fixture['units'], prior)
    first, second = key_arrays(scorer.inventory, fixture['mapping'], prior)
    n = len(scorer.inventory)
    pairs = np.asarray(list(combinations(range(n), 2)), dtype=np.int64)
    warm_started = time.monotonic()
    current = float(scorer.full(first, second)[0])
    warm = pairs[:min(8, len(pairs))]
    none = np.full(len(warm), -1)
    scorer.best(first, second, current, warm[:, 0], warm[:, 1], none, none, backend == 'incremental')
    warm_seconds = time.monotonic() - warm_started
    # Audit candidates far apart as well as adjacent/overlapping occurrences.
    rng = np.random.default_rng(20260925)
    chosen = np.unique(np.concatenate((np.arange(min(512, len(pairs))),
             np.arange(max(0, len(pairs)-512), len(pairs)),
             rng.choice(len(pairs), min(512, len(pairs)), replace=False))))
    audit = pairs[chosen]
    none = np.full(len(audit), -1)
    f, g = scorer.materialize(first, second, audit[:, 0], audit[:, 1], none, none)
    exact = scorer.full(f, g)
    approximate = scorer.approximate(first, current, audit[:, 0], audit[:, 1], none)
    max_error = float(np.max(np.abs(exact - approximate)))
    if max_error > scorer.tolerance:
        raise ArithmeticError('Representative score audit failed')
    del f, g, exact, approximate
    trials = []
    for _ in range(plan['repetitions']):
        gc.collect()
        start = time.monotonic()
        rescored = 0
        if backend == 'legacy':
            none = np.full(len(pairs), -1)
            f, g = scorer.materialize(first, second, pairs[:, 0], pairs[:, 1], none, none)
            scores = scorer.full(f, g)
            k = int(np.argmin(scores))
            best, best_index = float(scores[k]), k
            rescored = len(pairs)
        else:
            state = scorer.state(first) if backend == 'incremental' else None
            best, best_index, offset = float('inf'), None, 0
            for chunk in pair_batches(n, plan['batch_size']):
                none = np.full(len(chunk), -1)
                k, value, checked = scorer.best(first, second, current, chunk[:, 0], chunk[:, 1],
                                               none, none, backend == 'incremental', state)
                rescored += checked
                if value < best:
                    best, best_index = value, offset+k
                offset += len(chunk)
        seconds = time.monotonic() - start
        trials.append(dict(seconds=seconds, score=best, winner_index=best_index,
                           winner_pair=pairs[best_index].tolist(), full_rescores=rescored))
        if backend == 'legacy':
            del f, g, scores
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    return dict(name=name, backend=backend, units=n, positions=len(fixture['units']), pairs=len(pairs),
                warm_seconds=warm_seconds, trials=trials, median_seconds=float(np.median([r['seconds'] for r in trials])),
                peak_rss_bytes=rss, candidate_key_bytes=4*n*(len(pairs) if backend == 'legacy' else min(len(pairs),plan['batch_size'])),
                score_audit_candidates=len(audit), score_audit_max_error=max_error,
                numeric_tolerance=scorer.tolerance)


def run():
    plan = read(OUT / 'BENCHMARK_PLAN.json')
    if (OUT / 'benchmark.json').exists():
        raise FileExistsError('Benchmark already recorded')
    cases = fixtures()
    started = time.monotonic()
    rows = []
    env = dict(os.environ, NUMBA_NUM_THREADS='2', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for case in cases:
        for backend in plan['backends']:
            remaining = plan['max_wall_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Declared benchmark wall budget exhausted')
            completed = subprocess.run([sys.executable, '-m', 'experiments.rejection_refiner_benchmark', 'worker',
                                        '--name', case['name'], '--backend', backend], cwd=ROOT, env=env,
                                       capture_output=True, text=True, timeout=min(remaining,plan['worker_timeout_seconds']))
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            row = json.loads(completed.stdout)
            seal(STATE / 'benchmark' / f"{case['name']}-{backend}.json", row)
            rows.append(row)
            print(case['name'], backend, f"{row['median_seconds']:.3f}s", flush=True)
    for case in cases:
        group = [r for r in rows if r['name']==case['name']]
        reference = group[0]['trials'][0]
        for row in group:
            for trial in row['trials']:
                if any(trial[k] != reference[k] for k in ('score','winner_index','winner_pair')):
                    raise ArithmeticError('Backend winner differs from legacy')
    copying = {r['backend']:r for r in rows if r['name']=='english-copy'}
    selected = 'incremental' if copying['incremental']['median_seconds'] < copying['chunked']['median_seconds'] else 'chunked'
    seal(OUT / 'benchmark.json', dict(rows=rows, all_winners_identical=True, selected_backend=selected,
                                      measured_wall_seconds=time.monotonic()-started, threads=2))
    print('Selected backend:', selected, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker'])
    p.add_argument('--name');p.add_argument('--backend',choices=['legacy','chunked','incremental'])
    a=p.parse_args()
    if a.command=='run':run()
    else:print(json.dumps(worker(a.name,a.backend)))
