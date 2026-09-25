"""Meaning class from formula context: two predictors, balanced accuracy, and the two negative controls.

Both predictors see only where a type occurs: its neighbours and its position in the text.
Neither sees the letters of the type itself.
"""

import collections

import numpy as np

from etruscan.classes import CLASSES

K_NEIGHBOURS = 5
ITERATIONS = 10


def position(i, n):
    if n == 1:
        return "P:only"
    return "P:first" if i == 0 else "P:last" if i == n - 1 else "P:mid"


def _matrix(rows, types):
    """Sparse-free dense matrix from ``{type: Counter(feature)}`` (the corpora are small)."""
    features = sorted({f for t in types for f in rows[t]})
    index = {f: j for j, f in enumerate(features)}
    m = np.zeros((len(types), len(features)))
    for i, t in enumerate(types):
        for f, n in rows[t].items():
            m[i, index[f]] = n
    return m


def _cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def neighbour_words(texts, seeds):
    """M1: PPMI vectors over neighbour words and position; similarity-weighted 5-nearest seeds."""
    rows = collections.defaultdict(collections.Counter)
    for text in texts:
        for i, t in enumerate(text):
            rows[t]["L:" + (text[i - 1] if i else "<s>")] += 1
            rows[t]["R:" + (text[i + 1] if i < len(text) - 1 else "</s>")] += 1
            rows[t][position(i, len(text))] += 1
    types = sorted(rows)
    m = _matrix(rows, types)
    total = m.sum()
    expected = m.sum(1, keepdims=True) * m.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.where(m > 0, np.maximum(np.log(m / expected), 0), 0)
    seed_types = [t for t in types if t in seeds]
    seed_index = [types.index(t) for t in seed_types]
    sims = _cosine(ppmi, ppmi[seed_index])
    fallback = collections.Counter(seeds.values()).most_common(1)[0][0]
    out = {}
    for i, t in enumerate(types):
        if t in seeds:
            continue
        row = sims[i]
        top = np.argsort(-row)[:K_NEIGHBOURS]
        votes = collections.Counter()
        for j in top:
            if row[j] > 0:
                votes[seeds[seed_types[j]]] += row[j]
        out[t] = votes.most_common(1)[0][0] if votes else fallback
    return out


def neighbour_classes(texts, seeds):
    """M2: vectors over the classes of neighbours and position; nearest class centroid; 10 rounds."""
    types = sorted({t for text in texts for t in text})
    current = {}
    for _ in range(ITERATIONS):
        def cls(t):
            return seeds.get(t) or current.get(t) or "UNK"
        rows = collections.defaultdict(collections.Counter)
        for text in texts:
            for i, t in enumerate(text):
                rows[t]["LC:" + (cls(text[i - 1]) if i else "<s>")] += 1
                rows[t]["RC:" + (cls(text[i + 1]) if i < len(text) - 1 else "</s>")] += 1
                rows[t][position(i, len(text))] += 1
        m = _matrix(rows, types)
        m = m / m.sum(1, keepdims=True)
        labels = [c for c in CLASSES if any(v == c for v in seeds.values())]
        centroids = np.array([m[[i for i, t in enumerate(types) if seeds.get(t) == c]].mean(0) for c in labels])
        sims = _cosine(m, centroids)
        new = {t: labels[int(np.argmax(sims[i]))] for i, t in enumerate(types) if t not in seeds}
        if new == current:
            break
        current = new
    return current


METHODS = {"M1_neighbour_words": neighbour_words, "M2_neighbour_classes": neighbour_classes}


def balanced_accuracy(predicted, gold):
    recalls = []
    for c in CLASSES:
        items = [t for t, g in gold.items() if g == c]
        if items:
            recalls.append(sum(predicted.get(t) == c for t in items) / len(items))
    return float(np.mean(recalls)), len(recalls)


def split(labels, rng, share=0.2):
    """Stratified: ``share`` of each class held out, at least one per class."""
    seeds, held = {}, {}
    for c in CLASSES:
        items = sorted(t for t, v in labels.items() if v == c)
        rng.shuffle(items)
        k = max(1, round(share * len(items))) if items else 0
        held.update({t: c for t in items[:k]})
        seeds.update({t: c for t in items[k:]})
    return seeds, held


def shuffled(texts, rng):
    out = []
    for text in texts:
        text = list(text)
        rng.shuffle(text)
        out.append(text)
    return out


def corpus_shuffled(texts, rng):
    """All tokens pooled and redealt into texts of the same lengths: frequencies kept, context destroyed."""
    pool = [t for text in texts for t in text]
    rng.shuffle(pool)
    out, i = [], 0
    for text in texts:
        out.append(pool[i:i + len(text)])
        i += len(text)
    return out


def permuted(seeds, rng):
    keys = sorted(seeds)
    values = [seeds[k] for k in keys]
    rng.shuffle(values)
    return dict(zip(keys, values))


def replicate(texts, labels, rng):
    """One split: real, corpus-shuffled, permuted-label and (ungated) within-text-shuffled balanced accuracy."""
    seeds, held = split(labels, rng)
    order = shuffled(texts, rng)
    pooled = corpus_shuffled(texts, rng)
    wrong = permuted(seeds, rng)
    out = {"seeds": len(seeds), "held_out": len(held),
           "held_out_by_class": dict(collections.Counter(held.values()))}
    for name, method in METHODS.items():
        out[name] = {
            "real": balanced_accuracy(method(texts, seeds), held)[0],
            "corpus_shuffled": balanced_accuracy(method(pooled, seeds), held)[0],
            "permuted_labels": balanced_accuracy(method(texts, wrong), held)[0],
            "within_text_shuffled": balanced_accuracy(method(order, seeds), held)[0],
        }
    return out
