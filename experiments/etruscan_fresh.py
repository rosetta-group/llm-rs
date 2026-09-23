"""Etruscan round two: round-one gate re-run on deduplicated texts, then the fresh Wiktionary test.

    python -m experiments.etruscan_fresh

See experiments/etruscan-fresh/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from etruscan import classes, context, corpus, wiktionary
from experiments.etruscan_round_one import CHANCE, REPLICATES, latin_draw, passes, summarise
from linear_a.controls import rng_for

OUT = Path("experiments/etruscan-fresh/results.json")
NULLS = 1000


def texts(with_ciep):
    rows = corpus.load_rows().drop_duplicates(subset=["ID", "Etruscan", "key"])
    kept, dropped = corpus.dedupe_within_id(corpus.texts(rows, numerals=True))
    out = []
    for source, _, words in kept:
        if source == "ETP":
            out.append(words)
        elif with_ciep and len(words) >= 2 and not any(
                corpus.damaged(w) or len(w) >= 12 or set(w) & set("obdg") for w in words):
            out.append(words)
    return out, dropped


def etp_labels(ts):
    glossed = corpus.glossed_words()
    return {t: c for t in {t for text in ts for t in text} if (c := classes.etruscan_label(t, glossed))}


def gate_rerun(etp, labels):
    share = len(labels) / len({t for text in etp for t in text})
    lengths = [len(t) for t in etp]
    pool = collections.defaultdict(list)
    for words in classes.latin_epitaphs():
        pool[len(words)].append(words)
    latin = [context.replicate(*latin_draw(pool, lengths, share, rng_for("latin", r)), rng_for("latin-split", r))
             for r in range(REPLICATES)]
    etruscan = [context.replicate(etp, labels, rng_for("etruscan", r)) for r in range(REPLICATES)]
    ls, es = summarise(latin), summarise(etruscan)
    m2 = "M2_neighbour_classes"
    return {"latin": ls, "etruscan": es,
            "M2_passed": passes(ls[m2]) and passes(es[m2]),
            "size": {"texts": len(etp), "tokens": sum(lengths), "labelled_types": len(labels)}}


def score(predicted, gold):
    present = [c for c in classes.CLASSES if c in gold.values()]
    recall = {c: np.mean([predicted.get(t) == c for t, g in gold.items() if g == c]) for c in present}
    calls = {c: [t for t in gold if predicted.get(t) == c] for c in classes.CLASSES}
    precision = {c: (sum(gold[t] == c for t in ts) / len(ts) if ts else None) for c, ts in calls.items()}
    return {"balanced_accuracy": float(np.mean(list(recall.values()))),
            "accuracy": float(np.mean([predicted.get(t) == g for t, g in gold.items()])),
            "chance": 1 / len(present), "recall": {c: float(v) for c, v in recall.items()},
            "precision": precision}


def fresh(ts, seeds, gold, tag):
    method = context.neighbour_classes
    real = score(method(ts, seeds), gold)
    shuffled = [score(method(context.corpus_shuffled(ts, rng_for(tag, "corpus", i)), seeds), gold)["balanced_accuracy"]
                for i in range(NULLS)]
    permuted = [score(method(ts, context.permuted(seeds, rng_for(tag, "labels", i))), gold)["balanced_accuracy"]
                for i in range(NULLS)]
    b = real["balanced_accuracy"]
    real["p_corpus_shuffled"] = (1 + sum(x >= b for x in shuffled)) / (NULLS + 1)
    real["p_permuted_labels"] = (1 + sum(x >= b for x in permuted)) / (NULLS + 1)
    real["null_means"] = {"corpus_shuffled": float(np.mean(shuffled)), "permuted_labels": float(np.mean(permuted))}
    real["passed"] = bool(b >= real["chance"] + 0.20 and real["p_corpus_shuffled"] < 0.01
                          and real["p_permuted_labels"] < 0.01)
    real["items"] = len(gold)
    real["items_by_class"] = dict(collections.Counter(gold.values()))
    real["predictions"] = {t: [gold[t], p] for t, p in sorted(method(ts, seeds).items()) if t in gold}
    return real


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; refusing to overwrite")
    wk = wiktionary.labels()
    etp, dropped = texts(with_ciep=False)
    labels = etp_labels(etp)
    result = {"dedupe_dropped_texts": dropped, "gate_rerun": gate_rerun(etp, labels)}
    for name, with_ciep in (("primary_etp_plus_ciep", True), ("secondary_etp", False)):
        ts = texts(with_ciep)[0]
        seeds = etp_labels(ts)
        gold = {t: wk[t] for t in {t for text in ts for t in text} if t in wk and t not in seeds}
        result[name] = fresh(ts, seeds, gold, name)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (v if k != "gate_rerun" else {x: v[x] for x in ("M2_passed", "size", "etruscan", "latin")})
                      for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
