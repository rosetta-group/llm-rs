"""Etruscan round one, exploratory (after the result): per-class recall of M2 on the same 20 splits.

    python -m experiments.etruscan_round_one_classes

Not in the protocol. Rebuilds the identical splits (the split is the first draw from each
replicate's generator) and records which classes carry M2's balanced accuracy.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from etruscan import classes, context
from experiments.etruscan_round_one import REPLICATES, etruscan_labels, etruscan_texts, latin_draw
from linear_a.controls import rng_for

OUT = Path("experiments/etruscan/round-one-classes.json")


def recalls(texts, labels, rng):
    seeds, held = context.split(labels, rng)
    predicted = context.neighbour_classes(texts, seeds)
    out, confusion = {}, collections.Counter()
    for c in classes.CLASSES:
        items = [t for t, g in held.items() if g == c]
        out[c] = sum(predicted.get(t) == c for t in items) / len(items) if items else None
        confusion.update((c, predicted.get(t)) for t in items)
    return out, confusion


def summarise(runs):
    mean = {c: round(float(np.mean([r[c] for r, _ in runs if r[c] is not None])), 3) for c in classes.CLASSES}
    total = collections.Counter()
    for _, conf in runs:
        total.update(conf)
    confusion = {g: {p: total[(g, p)] for p in classes.CLASSES if total[(g, p)]} for g in classes.CLASSES}
    return {"recall": mean, "confusion_gold_to_predicted": confusion}


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; refusing to overwrite")
    etp = etruscan_texts()
    labels = etruscan_labels(etp)
    share = len(labels) / len({t for text in etp for t in text})
    pool = collections.defaultdict(list)
    for words in classes.latin_epitaphs():
        pool[len(words)].append(words)
    lengths = [len(t) for t in etp]
    latin = [recalls(*latin_draw(pool, lengths, share, rng_for("latin", r)), rng_for("latin-split", r))
             for r in range(REPLICATES)]
    etruscan = [recalls(etp, labels, rng_for("etruscan", r)) for r in range(REPLICATES)]
    result = {"method": "M2_neighbour_classes", "latin": summarise(latin), "etruscan": summarise(etruscan)}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
