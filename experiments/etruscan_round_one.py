"""Etruscan round one: does formula context carry meaning class? Latin and Etruscan controls, gated.

    python -m experiments.etruscan_round_one

See experiments/etruscan/PROTOCOL.md. Writes round-one-results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from etruscan import classes, context, corpus
from linear_a.controls import rng_for

OUT = Path("experiments/etruscan/round-one-results.json")
REPLICATES = 20
CHANCE = 1 / len(classes.CLASSES)
ANCHORS = {"clan": "KIN", "sec": "KIN", "puia": "KIN", "avils": "LIFE", "lupu": "LIFE",
           "svalce": "LIFE", "ril": "LIFE", "suthi": "LIFE", "ci": "NUM", "zal": "NUM"}


def etruscan_texts(with_ciep=False):
    rows = corpus.load_rows().drop_duplicates(subset=["ID", "Etruscan", "key"])
    out = []
    for source, _, words in corpus.texts(rows, numerals=True):
        if source == "ETP":
            out.append(words)
        elif with_ciep and len(words) >= 2 and not any(
                corpus.damaged(w) or len(w) >= 12 or set(w) & set("obdg") for w in words):
            out.append(words)
    return out


def etruscan_labels(texts):
    glossed = corpus.glossed_words()
    types = {t for text in texts for t in text}
    return {t: c for t in types if (c := classes.etruscan_label(t, glossed))}


def latin_draw(pool_by_length, lengths, labelled_share, rng):
    """Latin epitaphs with exactly the Etruscan length distribution; labels hidden at the Etruscan rate."""
    texts = []
    used = collections.defaultdict(set)
    for n in lengths:
        # No Latin epitaph of this length (one ETP text of 208 tokens): shortest longer one, truncated.
        m = n if pool_by_length[n] else min(k for k in pool_by_length if k > n)
        pool = pool_by_length[m]
        while True:
            j = int(rng.integers(len(pool)))
            if j not in used[m]:
                used[m].add(j)
                break
        texts.append([classes.latin_type(w) for w in pool[j][:n]])
    labels = {}
    for t in sorted({t for text in texts for t in text}):
        raw = t[2:].upper() if t.startswith("N:") else t
        c = classes.latin_label(raw)
        if c and rng.random() < labelled_share:
            labels[t] = c
    return texts, labels


def summarise(reps):
    out = {}
    for name in context.METHODS:
        out[name] = {k: round(float(np.mean([r[name][k] for r in reps])), 4)
                     for k in ("real", "corpus_shuffled", "permuted_labels", "within_text_shuffled")}
    return out


def passes(s):
    return (s["real"] >= CHANCE + 0.20 and s["corpus_shuffled"] <= CHANCE + 0.05
            and s["permuted_labels"] <= CHANCE + 0.05)


def anchors(texts, labels):
    out = {}
    for word, expected in ANCHORS.items():
        if word not in labels:
            out[word] = "absent"
            continue
        seeds = {t: c for t, c in labels.items() if t != word}
        out[word] = {name: method(texts, seeds).get(word) for name, method in context.METHODS.items()}
        out[word]["expected"] = expected
    return out


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; refusing to overwrite")
    etp = etruscan_texts()
    etp_labels = etruscan_labels(etp)
    types = {t for text in etp for t in text}
    share = len(etp_labels) / len(types)
    lengths = [len(t) for t in etp]

    pool_by_length = collections.defaultdict(list)
    for words in classes.latin_epitaphs():
        pool_by_length[len(words)].append(words)

    latin = [context.replicate(*latin_draw(pool_by_length, lengths, share, rng_for("latin", r)), rng_for("latin-split", r))
             for r in range(REPLICATES)]
    etruscan = [context.replicate(etp, etp_labels, rng_for("etruscan", r)) for r in range(REPLICATES)]
    ls, es = summarise(latin), summarise(etruscan)
    gate = {name: {"latin": passes(ls[name]), "etruscan": passes(es[name]),
                   "passed": passes(ls[name]) and passes(es[name])} for name in context.METHODS}

    both = etruscan_texts(with_ciep=True)
    both_labels = etruscan_labels(both)
    secondary = [context.replicate(both, both_labels, rng_for("etp+ciep", r)) for r in range(REPLICATES)]

    result = {
        "chance": CHANCE,
        "etruscan_size": {"texts": len(etp), "tokens": sum(lengths), "types": len(types),
                          "labelled_types": len(etp_labels),
                          "by_class": dict(collections.Counter(etp_labels.values()))},
        "latin": {"summary": ls, "replicates": latin},
        "etruscan": {"summary": es, "replicates": etruscan},
        "gate": gate,
        "anchors": anchors(etp, etp_labels),
        "secondary_etp_plus_ciep": {
            "size": {"texts": len(both), "tokens": sum(map(len, both)), "labelled_types": len(both_labels)},
            "summary": summarise(secondary), "replicates": secondary},
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("etruscan_size", "gate", "anchors")}, indent=1))
    print(json.dumps({"latin": ls, "etruscan": es, "secondary": result["secondary_etp_plus_ciep"]["summary"]}, indent=1))


if __name__ == "__main__":
    main()
