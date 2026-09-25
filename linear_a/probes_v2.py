"""Repairs for Linear A profiles and Monte Carlo tests; historical probes stay frozen."""

import collections
import math

import numpy as np

from linear_a import probes


def require_resolution(repeats, alpha):
    """Reject a simulation budget that cannot attain a strict p < alpha."""
    minimum = 1 / (repeats + 1) if repeats >= 1 else 1.
    if (repeats < 1 or not 0 < alpha < 1 or minimum >= alpha
            or math.isclose(minimum, alpha, rel_tol=1e-12)):
        raise ValueError(f"{repeats} null runs cannot attain p < {alpha}")


def monte_carlo(observed, values, alpha):
    values = np.asarray(values)
    require_resolution(len(values), alpha)
    exceedances = int(np.count_nonzero(values >= observed))
    # Wilson interval describes simulation uncertainty in the exceedance probability.
    n, z = len(values), 1.959963984540054
    rate = exceedances / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"observed": int(observed), "null_runs": n, "null_mean": float(values.mean()),
            "exceedances": exceedances, "p": probes.upper_p(observed, values),
            "minimum_p": 1 / (n + 1), "alpha": alpha,
            "below_alpha": probes.upper_p(observed, values) < alpha,
            "exceedance_probability_wilson95": [max(0., centre-radius), min(1., centre+radius)]}


def profile(words):
    """Type-level profile: identical occurrences are never different word forms."""
    words = sorted(set(w for w in words if len(w) >= 2))
    if not words:
        raise ValueError("profile needs at least one word of two or more signs")
    return probes.profile(words)


def by_length(words):
    buckets = collections.defaultdict(list)
    for w in sorted(set(words)):
        buckets[len(w)].append(w)
    return buckets


def common_quotas(pools, size, rng):
    """Draw exact-length quotas from capacities shared by every pool, without replacement.

    This explicitly conditions the comparison on common length support. It does not silently
    replace absent lengths with the most common bucket, or duplicate scarce reference types.
    """
    buckets = [by_length(p) for p in pools]
    lengths = sorted(set.intersection(*(set(b) for b in buckets)))
    slots = [n for n in lengths for _ in range(min(len(b[n]) for b in buckets))]
    if not slots:
        raise ValueError("pools have no common word lengths")
    selected = rng.choice(len(slots), min(size, len(slots)), replace=False)
    return collections.Counter(slots[i] for i in selected)


def matched_unique(words, quotas, rng):
    buckets = by_length(words)
    out = []
    for n, count in sorted(quotas.items()):
        if len(buckets[n]) < count:
            raise ValueError(f"length {n}: need {count} unique types; have {len(buckets[n])}")
        out.extend(buckets[n][i] for i in rng.choice(len(buckets[n]), count, replace=False))
    return out


def nearest_profiles(target_samples, reference_samples):
    # Fit the scale on references only: target profiles cannot change the metric.
    reference = np.vstack([p for ps in reference_samples.values() for p in ps])
    scale = reference.std(axis=0)
    scale[scale == 0] = 1.
    centres = {k: np.mean(v, axis=0) for k, v in reference_samples.items()}
    distances = [{k: float(np.linalg.norm((p-c) / scale)) for k, c in centres.items()}
                 for p in target_samples]
    return [min(d, key=d.get) for d in distances], distances


class SubstitutionIndex:
    """Exact one-sign pairs, indexed by the unchanged portion; same counts as v1."""

    def __init__(self, words):
        self.index = collections.defaultdict(list)
        for w in words:
            if len(w) >= 3:
                for i in range(len(w)):
                    self.index[len(w), i, w[:i] + w[i+1:]].append(w)

    def count(self, words):
        counts = collections.Counter()
        for w in words:
            if len(w) < 3:
                continue
            for i, old in enumerate(w):
                pos = "final" if i == len(w)-1 else "initial" if i == 0 else "medial"
                for v in self.index.get((len(w), i, w[:i] + w[i+1:]), ()):
                    if v[i] != old:
                        counts[pos, old, v[i]] += 1
        return counts
