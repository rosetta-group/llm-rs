"""Joint blind search for numeral values under K latent number systems.

Values are exact: every grid value is scaled by SCALE = 2^5 3^3 5^3, so reciprocals down to 1/60
are integers and a tablet balances when an integer dot product is exactly zero.
"""

from fractions import Fraction
from itertools import product

import numpy as np

SCALE = 2**5 * 3**3 * 5**3


def grid():
    values = {2**a * 3**b * 5**c for a, b, c in product(range(6), range(4), range(4))} - {1}
    fractions = {Fraction(1, v) for v in values if v <= 60}
    return sorted([Fraction(v) for v in values] + list(fractions))


GRID = grid()
GRID_SCALED = np.array([int(g * SCALE) for g in GRID], dtype=np.int64)


def matrix(tablets, units):
    index = {u: i for i, u in enumerate(units)}
    d = np.zeros((len(tablets), len(units)), dtype=np.int64)
    for row, (sums, total) in enumerate(tablets):
        for u, c in sums.items():
            d[row, index[u]] += c
        for u, c in total.items():
            d[row, index[u]] -= c
    return d


def score(d, values):
    return int(np.any(d @ values.T == 0, axis=1).sum())


def climb(d, base, k, generator, max_steps=200):
    """One restart of greedy coordinate ascent. ``base`` is the column of N01 (fixed at 1)."""
    t, u = d.shape
    values = GRID_SCALED[generator.integers(len(GRID_SCALED), size=(k, u))]
    values[:, base] = SCALE
    products = d @ values.T
    best = int(np.any(products == 0, axis=1).sum())
    for _ in range(max_steps):
        candidates = []
        for system in range(k):
            others = np.delete(products, system, axis=1)
            balanced_other = np.any(others == 0, axis=1) if k > 1 else np.zeros(t, bool)
            for unit in range(u):
                if unit == base:
                    continue
                column = products[:, system][:, None] + d[:, unit][:, None] * (
                    GRID_SCALED[None, :] - values[system, unit])
                scores = (balanced_other[:, None] | (column == 0)).sum(axis=0)
                top = scores.max()
                if top > best:
                    choices = np.flatnonzero(scores == top)
                    candidates.append((int(top), system, unit, int(generator.choice(choices))))
        if not candidates:
            break
        top = max(c[0] for c in candidates)
        pick = [c for c in candidates if c[0] == top]
        _, system, unit, g = pick[generator.integers(len(pick))]
        products[:, system] += d[:, unit] * (GRID_SCALED[g] - values[system, unit])
        values[system, unit] = GRID_SCALED[g]
        best = top
    return best, values


def search(d, base, k, restarts, generator):
    best = (-1, None)
    for _ in range(restarts):
        s, v = climb(d, base, k, generator)
        if s > best[0]:
            best = (s, v)
    return best


def describe(d, values, units, names=None):
    """Per system: tablets it balances (first system wins) and values of the units it determines."""
    products = d @ values.T
    owner = np.where(np.any(products == 0, axis=1), np.argmax(products == 0, axis=1), -1)
    systems = []
    for k in range(values.shape[0]):
        rows = np.flatnonzero(owner == k)
        determined = [i for i in range(len(units)) if np.any(d[rows, i] != 0)] if len(rows) else []
        systems.append({
            "tablets": int(len(rows)),
            "values": {units[i]: str(Fraction(int(values[k, i]), SCALE)) for i in determined},
            "examples": [names[r] for r in rows[:6]] if names else [],
        })
    return sorted(systems, key=lambda s: -s["tablets"])
