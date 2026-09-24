"""Proto-Elamite round one: blind numeral ratios from balanced two-sign tablets.

    python -m experiments.proto_elamite_round_one

See experiments/proto-elamite/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from linear_a.controls import rng_for
from proto_elamite import numerals as N

OUT = Path("experiments/proto-elamite")
NULL_RUNS = 1000
MIN_TABLETS = 5
SUBSAMPLES = 20


def pair_data(tablets, before):
    """``{(big, small): [(entry_sum, total), ...]}`` for usable two-sign tablets."""
    out = collections.defaultdict(list)
    for t in tablets:
        if not t["usable"]:
            continue
        pair = N.two_sign(t)
        if pair:
            big, small = N.larger(*pair, before)
            out[(big, small)].append((N.entry_sum(t), t["total"], t["designation"]))
    return out


def supports(items, big, small):
    return collections.Counter(r for s, total, _ in items
                               if (r := N.candidate(s, total, big, small)) is not None)


def null_max(items, big, small, generator, runs=NULL_RUNS):
    sums = [s for s, _, _ in items]
    totals = [t for _, t, _ in items]
    n = len(items)
    out = []
    for _ in range(runs):
        perm = generator.permutation(n)
        while n > 1 and np.any(perm == np.arange(n)):
            perm = generator.permutation(n)
        c = collections.Counter(r for i in range(n)
                                if (r := N.candidate(sums[i], totals[perm[i]], big, small)) is not None)
        out.append(max(c.values()) if c else 0)
    return out


def analyse(items, big, small, tag):
    support = supports(items, big, small)
    ranked = support.most_common()
    null = null_max(items, big, small, rng_for("proto-elamite", tag, big, small))
    top = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0
    p = lambda k: float((np.sum(np.array(null) >= k) + 1) / (len(null) + 1))
    return {"pair": f"{big}/{small}", "tablets": len(items), "support": dict(ranked[:6]),
            "top": ranked[0][0] if ranked else None, "p_top": p(top),
            "second": ranked[1][0] if len(ranked) > 1 else None, "p_second": p(second),
            "null_max_mean": float(np.mean(null)),
            "examples": {str(r): [d for s, t, d in items if N.candidate(s, t, big, small) == r][:5]
                         for r, _ in ranked[:3]}}


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    control = N.tablets("uruk-iii") + N.tablets("uruk-iv")
    target = N.tablets("proto-elamite")
    counts = {name: {"administrative": len(ts), "clean": sum(t["clean"] for t in ts),
                     "usable": sum(t["usable"] for t in ts)}
              for name, ts in (("proto-cuneiform", control), ("proto-elamite", target))}
    print(counts, flush=True)
    control_pairs = pair_data(control, N.precedence(control))
    target_pairs = pair_data(target, N.precedence(target))

    key = ("N14", "N01")
    full = analyse(control_pairs[key], *key, "control-full")
    print("control full", full, flush=True)
    full_pass = {full["top"], full["second"]} == {10, 6} and max(full["p_top"], full["p_second"]) < 0.01
    size = len(target_pairs.get(key, []))
    sub_hits = []
    generator = rng_for("proto-elamite", "subsamples")
    pool = control_pairs[key]
    for d in range(SUBSAMPLES):
        pick = [pool[i] for i in generator.choice(len(pool), min(size, len(pool)), replace=False)]
        ranked = supports(pick, *key).most_common(2)
        sub_hits.append({r for r, _ in ranked} == {10, 6})
    gate = full_pass and sum(sub_hits) >= 18
    print("gate", full_pass, sum(sub_hits), "of", SUBSAMPLES, "at size", size, flush=True)

    other_control = [analyse(v, *k, "control") for k, v in sorted(control_pairs.items())
                     if len(v) >= MIN_TABLETS and k != key]
    result = {"counts": counts, "control_N14_N01": full, "control_subsample_size": size,
              "control_subsample_hits": int(sum(sub_hits)), "gate": gate,
              "control_other_pairs": other_control}
    if gate:
        result["target"] = [analyse(v, *k, "target") for k, v in sorted(target_pairs.items())
                            if len(v) >= MIN_TABLETS]
        for row in result["target"]:
            print("target", row["pair"], row["tablets"], row["support"], row["p_top"], flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
