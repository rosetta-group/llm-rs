"""Linear A round five: length-matched profiles against TLHdig languages; Peet's Keftiu names.

    python -m experiments.linear_a_tlhdig

See experiments/linear-a-tlhdig/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from experiments.linear_a_probes import _running_greek, linear_a, linear_b
from linear_a import probes, tlhdig
from linear_a.controls import rng_for
from linear_a.spelling import render

OUT = Path("experiments/linear-a-tlhdig")
SIZE = 300
SAMPLES = 20
POWER = 18
NAMES = {"Hit": "Hittite", "Luw": "Luwian", "Pal": "Palaic", "Hur": "Hurrian", "Hat": "Hattic",
         "Akk": "Akkadian"}
# Peet 1927, p. 92, consonants only; initial aleph / i, and h-dot, dropped (see protocol).
KEFTIU = {"ꜣšḥr": "sr", "Nsy": "ns", "ꜣkš": "ks", "ꜣkšt": "kst", "ꜣdm": "tm", "Pnrt": "pnrt",
          "Rs": "rs", "Bndbr": "pntpr", "ỉknw": "knw"}


def spelled_lists():
    lists = {"Greek": _running_greek()}
    for code, words in tlhdig.forms().items():
        spelled = {probes_spell(w) for w in words}
        lists[NAMES[code]] = sorted(s for s in spelled if s and len(s) >= 2)
    return lists


def probes_spell(word):
    from linear_a.spelling import spell
    return spell(word)


def by_length(words):
    out = collections.defaultdict(list)
    for w in words:
        out[min(len(w), 6)].append(w)
    return out


def matched(pool, lengths, generator):
    buckets = by_length(pool)
    out = []
    for n in lengths:
        bucket = buckets.get(min(n, 6)) or max(buckets.values(), key=len)
        out.append(bucket[generator.integers(len(bucket))])
    return out


def nearest_counts(target, references, tag, size=SIZE):
    generator = rng_for("tlhdig", tag)
    size = min(size, len(target))
    target_profiles, reference_profiles = [], {k: [] for k in references}
    for _ in range(SAMPLES):
        sample = [target[i] for i in generator.choice(len(target), size, replace=False)]
        lengths = [len(w) for w in sample]
        target_profiles.append(probes.profile(sample))
        for name, pool in references.items():
            reference_profiles[name].append(probes.profile(matched(pool, lengths, generator)))
    nearest, _ = probes.nearest_profiles(target_profiles, reference_profiles)
    return dict(collections.Counter(nearest)), size


def keftiu():
    la_words = sorted(linear_a()["labels"])
    hits = {name: [render(w) for w in la_words
                   if probes.skeleton_matches(sk, probes.word_skeleton(w))]
            for name, sk in KEFTIU.items()}
    observed = sum(bool(v) for v in hits.values())
    null = []
    for corpus in probes.null_corpora(la_words, rng_for("tlhdig", "keftiu"), 200):
        null.append(sum(any(probes.skeleton_matches(sk, probes.word_skeleton(w)) for w in corpus)
                        for sk in KEFTIU.values()))
    return {"skeletons": KEFTIU, "hits": {k: v[:12] for k, v in hits.items()},
            "hit_counts": {k: len(v) for k, v in hits.items()},
            "names_with_hits": observed, "null_mean": float(np.mean(null)),
            "p": probes.upper_p(observed, null)}


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    lists = spelled_lists()
    sizes = {k: len(v) for k, v in lists.items()}
    print("list sizes", sizes, flush=True)

    control_1 = {}
    for name in NAMES.values():
        words = lists[name]
        halves = {h: [w for w in words if probes.half(w) == h] for h in (0, 1)}
        references = {**lists, name: halves[0]}
        counts, size = nearest_counts(halves[1], references, f"self-{name}")
        control_1[name] = {"nearest": counts, "sample_size": size,
                           "passed": counts.get(name, 0) >= POWER}
        print("control 1", name, counts, size, flush=True)

    lb_words = [w for w in linear_b()["labels"] if len(w) >= 2]
    control_2, _ = nearest_counts(lb_words, lists, "linear-b")
    control_2_passed = control_2.get("Greek", 0) >= POWER
    print("control 2", control_2, flush=True)

    la_words = [w for w in linear_a()["labels"] if len(w) >= 2]
    la_counts, _ = nearest_counts(la_words, lists, "linear-a")
    la_entry = [w for w, l in linear_a()["labels"].items() if l == "entry"]
    la_entry_counts, _ = nearest_counts(la_entry, lists, "linear-a-entry")
    print("linear a", la_counts, "entry", la_entry_counts, flush=True)
    top, top_n = max(la_counts.items(), key=lambda kv: kv[1])
    lead = (control_2_passed and top_n >= POWER and control_1.get(top, {}).get("passed", False))

    result = {"list_sizes": sizes, "control_1": control_1, "control_2": control_2,
              "control_2_passed": control_2_passed, "linear_a": la_counts,
              "linear_a_entry": la_entry_counts, "lead": top if lead else None,
              "keftiu": keftiu()}
    print("keftiu", {k: result["keftiu"][k] for k in ("names_with_hits", "null_mean", "p")})
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
