"""Bounded refinement with chunked exact scoring and one-letter incremental proposals.

Old frozen callers keep using variable_units.refine. This module preserves its
objective and candidate order, rechecks close incremental minima with the old full
scorer, and reports work/time limits separately from convergence.
"""
from itertools import combinations, islice
import math
import time

import numpy as np
from numba import njit, prange

from voynich.variable_units import _prepare, key_arrays, variable_scores


def pair_batches(size, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    pairs = combinations(range(size), 2)
    while True:
        chunk = list(islice(pairs, batch_size))
        if not chunk:
            return
        yield np.asarray(chunk, dtype=np.int64)


@njit(cache=True)
def _window_costs(key, values, costs, offsets, base, order):
    out = np.empty(len(values), dtype=np.float64)
    for end in range(len(values)):
        index = key[values[end]]
        power = base
        for k in range(1, min(end, order - 1) + 1):
            index += key[values[end - k]] * power
            power *= base
        out[end] = costs[offsets[min(end, order - 1)] + index]
    return out


@njit(cache=True, parallel=True)
def incremental_scores(key, values, positions, starts, old_costs, units_per, occ_per,
                       costs, offsets, base, order, current, targets, partners, replacements):
    """Approximate full scores; only changed windows and homophone groups differ.

    partners=-1 means a one-letter replacement. Otherwise swap two one-letter
    expansions. No mutable per-proposal copy of the text or complete key is needed.
    """
    out = np.empty(len(targets), dtype=np.float64)
    n = len(values)
    for h in prange(len(targets)):
        u, v = targets[h], partners[h]
        a = key[u]
        b = key[v] if v >= 0 else replacements[h]
        if a == b:
            out[h] = current
            continue
        q, q_end = starts[u], starts[u + 1]
        r = starts[v] if v >= 0 else 0
        r_end = starts[v + 1] if v >= 0 else 0
        last_end = -1
        delta = 0.
        # Merge sorted occurrence lists and visit the union of affected windows.
        while q < q_end or r < r_end:
            if r == r_end or (q < q_end and positions[q] < positions[r]):
                p = positions[q]
                q += 1
            else:
                p = positions[r]
                r += 1
            end_exclusive = min(n, p + order)
            for end in range(max(p, last_end + 1), end_exclusive):
                index = 0
                power = 1
                for k in range(min(end, order - 1) + 1):
                    unit = values[end - k]
                    letter = b if unit == u else (a if unit == v else key[unit])
                    index += letter * power
                    power *= base
                delta += costs[offsets[min(end, order - 1)] + index] - old_costs[end]
            last_end = max(last_end, end_exclusive - 1)
        fu = q_end - starts[u]
        if v >= 0:
            fv = r_end - starts[v]
            delta += (fv - fu) * (np.log2(units_per[a]) - np.log2(units_per[b]))
        else:
            ca, cb = units_per[a], units_per[b]
            oa, ob = occ_per[a], occ_per[b]
            if ca > 1:
                delta -= oa * np.log2(ca)
            if cb > 1:
                delta -= ob * np.log2(cb)
            if ca > 2:
                delta += (oa - fu) * np.log2(ca - 1)
            if cb > 0:
                delta += (ob + fu) * np.log2(cb + 1)
        out[h] = current + delta
    return out


class Scorer:
    def __init__(self, units, prior):
        (self.inventory, self.values, self.freq, self.costs, self.offsets,
         self.sb, self.db) = _prepare(units, prior)
        self.base, self.order = prior.base, prior.order
        self.positions = np.argsort(self.values, kind='stable').astype(np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.freq))).astype(np.int64)
        # Conservative accumulation-error allowance; near minima use exact scores.
        n, u = len(self.values), len(self.inventory)
        magnitude = n * float(np.max(np.abs(self.costs))) + u * self.db + n * math.log2(max(2, u))
        self.tolerance = 64 * np.finfo(np.float64).eps * (n + u + self.base + 1) * max(1., magnitude)

    def full(self, first, second):
        return variable_scores(first, second, self.values, self.freq, self.costs,
                               self.offsets, self.base, self.order, self.sb, self.db)

    def state(self, first):
        key = first[0]
        windows = _window_costs(key, self.values, self.costs, self.offsets, self.base, self.order)
        units_per = np.bincount(key, minlength=self.base).astype(np.int64)
        occ_per = np.bincount(key, weights=self.freq, minlength=self.base).astype(np.int64)
        return windows, units_per, occ_per

    def approximate(self, first, current, targets, partners, replacements, state=None):
        windows, units_per, occ_per = self.state(first) if state is None else state
        return incremental_scores(first[0], self.values, self.positions, self.starts,
                                  windows, units_per, occ_per, self.costs, self.offsets,
                                  self.base, self.order, current, targets, partners, replacements)

    @staticmethod
    def materialize(first, second, targets, partners, replacements, replacement_second):
        f = np.repeat(first, len(targets), axis=0)
        g = np.repeat(second, len(targets), axis=0)
        rows = np.arange(len(targets))
        swap = partners >= 0
        f[rows, targets] = np.where(swap, first[0, np.maximum(partners, 0)], replacements)
        g[rows, targets] = np.where(swap, second[0, np.maximum(partners, 0)], replacement_second)
        if np.any(swap):
            f[rows[swap], partners[swap]] = first[0, targets[swap]]
            g[rows[swap], partners[swap]] = second[0, targets[swap]]
        return f, g

    def best(self, first, second, current, targets, partners, replacements, replacement_second,
             incremental=False, state=None):
        """Earliest full-score minimum; incremental scores only shortlist candidates."""
        if not incremental:
            f, g = self.materialize(first, second, targets, partners, replacements, replacement_second)
            scores = self.full(f, g)
            k = int(np.argmin(scores))
            return k, float(scores[k]), len(scores)
        if np.any(second >= 0) or np.any(replacement_second >= 0):
            raise ValueError('Incremental scorer requires one-letter expansions')
        approximate = self.approximate(first, current, targets, partners, replacements, state)
        keep = np.flatnonzero(approximate <= approximate.min() + 2 * self.tolerance)
        scores = np.full(len(keep), current, dtype=np.float64)
        # Swapping identical readings and retaining the same letter are exact no-ops.
        changed = first[0, targets[keep]] != np.where(partners[keep] >= 0,
                    first[0, np.maximum(partners[keep], 0)], replacements[keep])
        selected = keep[changed]
        if len(selected):
            f, g = self.materialize(first, second, targets[selected], partners[selected],
                                    replacements[selected], replacement_second[selected])
            exact = self.full(f, g)
            if np.any(np.abs(exact - approximate[selected]) > self.tolerance):
                raise ArithmeticError('Incremental score exceeds its numerical error allowance')
            scores[changed] = exact
        k = int(np.argmin(scores))
        return int(keep[k]), float(scores[k]), int(np.count_nonzero(changed))


def refine(units, prior, mapping, bigrams=(), sweeps=50, kicks=20, kick_size=6, seed=0,
           cap=300, batch_size=512, max_evaluations=20_000_000, backend='incremental'):
    if not units or batch_size < 1 or (sweeps is not None and sweeps < 1):
        raise ValueError('Nonempty units, positive batch size and positive sweep limit required')
    if max_evaluations is not None and max_evaluations < 0:
        raise ValueError('max_evaluations must be nonnegative or None')
    if backend not in ('chunked', 'incremental'):
        raise ValueError('Unknown scoring backend')
    started = time.monotonic()
    scorer = Scorer(units, prior)
    first, second = key_arrays(scorer.inventory, mapping, prior)
    letters = [(a, -1) for a in range(prior.base)] + list(bigrams)
    opt_a = np.array([a for a, _ in letters], dtype=np.int16)
    opt_b = np.array([b for _, b in letters], dtype=np.int16)
    incremental = backend == 'incremental' and not bigrams and not np.any(second >= 0)
    current = float(scorer.full(first, second)[0])
    best_first, best_second, best = first.copy(), second.copy(), current
    rng = np.random.default_rng(seed)
    size = len(scorer.inventory)
    kicks_done = evaluations = improvements = sweeps_done = full_rechecks = 0
    reason = None

    def time_up():
        return cap is not None and time.monotonic() - started > cap

    def can_evaluate(count):
        return max_evaluations is None or evaluations + count <= max_evaluations

    while reason is None:
        if time_up():
            reason = 'time_limit'
            break
        if sweeps is not None and sweeps_done >= sweeps:
            reason = 'sweep_limit'
            break
        sweeps_done += 1
        improved = False
        for unit in rng.permutation(size):
            if not can_evaluate(len(letters)):
                reason = 'evaluation_limit'
                break
            targets = np.full(len(letters), unit, dtype=np.int64)
            partners = np.full(len(letters), -1, dtype=np.int64)
            k, value, rescored = scorer.best(first, second, current, targets, partners,
                                            opt_a, opt_b, incremental)
            evaluations += len(letters)
            full_rechecks += rescored
            if value < current - 1e-9:
                first[0, unit], second[0, unit], current = opt_a[k], opt_b[k], value
                improved = True
                improvements += 1
            if time_up():
                reason = 'time_limit'
                break
        if reason is None:
            pair_count = size * (size - 1) // 2
            if not can_evaluate(pair_count):
                reason = 'evaluation_limit'
            else:
                best_pair, pair_score = None, float('inf')
                state = scorer.state(first) if incremental and pair_count else None
                for pairs in pair_batches(size, batch_size):
                    targets, partners = pairs[:, 0], pairs[:, 1]
                    none = np.full(len(pairs), -1, dtype=np.int16)
                    k, value, rescored = scorer.best(first, second, current, targets, partners,
                                                    none, none, incremental, state)
                    evaluations += len(pairs)
                    full_rechecks += rescored
                    # Strict comparison retains the earliest pair across batch boundaries.
                    if value < pair_score:
                        best_pair, pair_score = pairs[k].copy(), value
                    if time_up():
                        reason = 'time_limit'
                        break
                if reason is None and best_pair is not None and pair_score < current - 1e-9:
                    u, v = best_pair
                    first[0, u], first[0, v] = first[0, v], first[0, u]
                    second[0, u], second[0, v] = second[0, v], second[0, u]
                    current = pair_score
                    improved = True
                    improvements += 1
        if current < best - 1e-9:
            best_first, best_second, best = first.copy(), second.copy(), current
        if reason is not None:
            break
        if not improved and kicks_done >= kicks:
            reason = 'converged'
            break
        if not improved:
            kicks_done += 1
            first, second = best_first.copy(), best_second.copy()
            for unit in rng.choice(size, size=min(kick_size, size), replace=False):
                k = rng.integers(len(letters))
                first[0, unit], second[0, unit] = opt_a[k], opt_b[k]
            current = float(scorer.full(first, second)[0])
    result_mapping = {unit: prior.alphabet[a] + (prior.alphabet[b] if b >= 0 else '')
                      for unit, a, b in zip(scorer.inventory, best_first[0], best_second[0])}
    return dict(status='complete' if reason == 'converged' else 'budget_exhausted',
                mapping=result_mapping, recovered=''.join(result_mapping[u] for u in units),
                search_bits=best, seconds=time.monotonic() - started, kicks=kicks_done,
                evaluations=evaluations, improvements=improvements, sweeps=sweeps_done,
                full_rechecks=full_rechecks, stop_reason=reason, cap_hit=reason != 'converged',
                backend='incremental' if incremental else 'chunked', batch_size=batch_size,
                numeric_tolerance=scorer.tolerance)
