"""Proto-Elamite round two: joint blind recovery of number systems (K = 3), control first.

    python -m experiments.proto_elamite_round_two

See experiments/proto-elamite-joint/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from linear_a.controls import rng_for
from proto_elamite import joint
from proto_elamite import numerals as N

OUT = Path("experiments/proto-elamite-joint")
K, RESTARTS, NULL_RUNS, NULL_RESTARTS = 3, 200, 50, 20
SUBSAMPLES, SUB_RESTARTS, MIN_UNIT_TABLETS = 20, 20, 5


def informative(tablets):
    rows = []
    for t in tablets:
        if not t["usable"]:
            continue
        sums, total = N.entry_sum(t), t["total"]
        if any(sums[u] != total[u] for u in set(sums) | set(total)):
            rows.append((sums, total, t["designation"]))
    counts = collections.Counter(u for s, tot, _ in rows for u in set(s) | set(tot))
    units = sorted(u for u, c in counts.items() if c >= MIN_UNIT_TABLETS)
    if "N01" not in units:
        units = ["N01"] + units
    kept = [r for r in rows if set(r[0]) | set(r[1]) <= set(units)]
    return kept, units, len(rows) - len(kept)


def n14_values(d, values, units):
    systems = joint.describe(d, values, units)
    return [(Fraction(s["values"]["N14"]) if "N14" in s["values"] else None, s["tablets"]) for s in systems]


def null_scores(rows, units, tag, runs, restarts):
    generator = rng_for("pe-joint-null", tag)
    n = len(rows)
    out = []
    for _ in range(runs):
        perm = generator.permutation(n)
        while np.any(perm == np.arange(n)):
            perm = generator.permutation(n)
        shuffled = [(rows[i][0], rows[perm[i]][1]) for i in range(n)]
        d = joint.matrix(shuffled, units)
        out.append(joint.search(d, units.index("N01"), K, restarts, generator)[0])
    return out


def run(rows, units, tag, restarts=RESTARTS):
    d = joint.matrix([(s, t) for s, t, _ in rows], units)
    best, values = joint.search(d, units.index("N01"), K, restarts, rng_for("pe-joint", tag))
    return d, best, values


def gate_ok(pairs):
    found = {v for v, n in pairs if v is not None and n >= 5}
    return Fraction(10) in found and Fraction(6) in found


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    control_rows, control_units, control_set_aside = informative(N.tablets("uruk-iii") + N.tablets("uruk-iv"))
    target_rows, target_units, target_set_aside = informative(N.tablets("proto-elamite"))
    print("control", len(control_rows), control_units, "set aside", control_set_aside, flush=True)
    print("target", len(target_rows), target_units, "set aside", target_set_aside, flush=True)

    d, best, values = run(control_rows, control_units, "control")
    names = [r[2] for r in control_rows]
    systems = joint.describe(d, values, control_units, names)
    pairs = n14_values(d, values, control_units)
    null = null_scores(control_rows, control_units, "control", NULL_RUNS, NULL_RESTARTS)
    p = float((np.sum(np.array(null) >= best) + 1) / (len(null) + 1))
    print("control best", best, "null max", max(null), "p", p, flush=True)
    for s in systems:
        print("  system", s["tablets"], s["values"], s["examples"][:3], flush=True)
    full_pass = gate_ok(pairs) and max(null) < best

    size = len(target_rows)
    generator = rng_for("pe-joint", "subsamples")
    hits = []
    for i in range(SUBSAMPLES):
        pick = [control_rows[j] for j in generator.choice(len(control_rows), min(size, len(control_rows)), replace=False)]
        dd, b, v = run(pick, control_units, f"sub-{i}", SUB_RESTARTS)
        hits.append(gate_ok(n14_values(dd, v, control_units)))
    gate = full_pass and sum(hits) >= 18
    print("gate", full_pass, sum(hits), "of", SUBSAMPLES, "at size", size, flush=True)

    result = {"control": {"informative": len(control_rows), "units": control_units,
                          "set_aside": control_set_aside, "best": best, "null": null, "p": p,
                          "systems": systems, "subsample_size": size,
                          "subsample_hits": int(sum(hits))},
              "gate": gate}
    if gate:
        d, best, values = run(target_rows, target_units, "target")
        null = null_scores(target_rows, target_units, "target", NULL_RUNS, NULL_RESTARTS)
        result["target"] = {"informative": len(target_rows), "units": target_units,
                            "set_aside": target_set_aside, "best": best, "null": null,
                            "p": float((np.sum(np.array(null) >= best) + 1) / (len(null) + 1)),
                            "systems": joint.describe(d, values, target_units, [r[2] for r in target_rows])}
        print("target", result["target"]["best"], "p", result["target"]["p"], flush=True)
        for s in result["target"]["systems"]:
            print("  system", s["tablets"], s["values"], flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
