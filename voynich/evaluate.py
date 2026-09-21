"""Aggregate token sums; bootstrap independent manuscript groups."""

import math
import hashlib
import json
from collections import defaultdict

import numpy as np

from .data import cluster


def target_signature(text, positions=None):
    if positions is None:
        positions = [i for i, char in enumerate(text) if char != "?"]
    payload = json.dumps([text, positions], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def total(rows):
    n = sum(r["tokens"] for r in rows)
    units = sum(r["units"] for r in rows)
    loss = sum(r["nll"] for r in rows)
    return dict(tokens=n, units=units, nll=loss,
                loss_nats=loss/n if n else None,
                bits_per_character=loss/(units*math.log(2)) if units else None,
                accuracy=sum(r["correct"] for r in rows)/n if n else None)


def summarize(rows):
    result = {"overall": total(rows)}
    for field in ("currier", "section", "hand"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[field] = {key: total(value) for key, value in sorted(groups.items())}
    return result


def paired_interval(reference, candidate, group="folio", seed=42, draws=2000):
    """Positive delta means the candidate uses fewer bits per character."""
    if any(not row.get("target_sha256") for row in [*reference, *candidate]):
        raise ValueError("Scores need target fingerprints; regenerate older score files")
    ref = {r["page"]: r for r in reference}
    cand = {r["page"]: r for r in candidate}
    if len(ref) != len(reference) or len(cand) != len(candidate):
        raise ValueError("Duplicate page scores")
    if ref.keys() != cand.keys():
        raise ValueError("Paired scores must cover identical pages")
    aggregates = defaultdict(lambda: np.zeros(3))
    for page, row in ref.items():
        other = cand[page]
        if (row["units"] != other["units"] or row[group] != other[group]
                or row["target_sha256"] != other["target_sha256"]):
            raise ValueError("Paired scores must use identical targets and manuscript groups")
        aggregates[cluster(row[group], group)] += (row["nll"], other["nll"], row["units"])
    if len(aggregates) < 2:
        raise ValueError("At least two groups are required for uncertainty estimates")
    values = np.array(list(aggregates.values()))
    samples = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    sums = values[samples].sum(axis=1)
    delta = (sums[:, 0] - sums[:, 1]) / (sums[:, 2] * math.log(2))
    point = (values[:, 0].sum() - values[:, 1].sum()) / (values[:, 2].sum()*math.log(2))
    return dict(delta_bits_per_character=float(point),
                interval_95=[float(x) for x in np.quantile(delta, [.025, .975])],
                group=group, groups=len(values), draws=draws, seed=seed)
