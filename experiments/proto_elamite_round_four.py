"""Proto-Elamite round four: object signs by system, labels from leave-one-out fits.

    python -m experiments.proto_elamite_round_four

See experiments/proto-elamite-signs-loo/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

from experiments.proto_elamite_round_two import K, informative
from linear_a.controls import rng_for
from proto_elamite import joint
from proto_elamite import numerals as N
from proto_elamite import signs as S

OUT = Path("experiments/proto-elamite-signs-loo")
RESTARTS, RELIABLE = 20, 5
GRAIN = {"M288", "M036", "M297"}
BEINGS = {"M388", "M124", "M346", "M367", "M006", "M362", "M376"}


def sign_index(folders):
    out = {}
    for folder in folders:
        for record in N.records(folder):
            out[record.get("designation")] = S.tablet_signs((record.get("inscription") or {}).get("atf") or "")
    return out


def system_classes(d, values, units):
    """Class of each fitted system ('capacity', 'counting' or None) under the reliability bar."""
    products = d @ values.T
    owner = np.where(np.any(products == 0, axis=1), np.argmax(products == 0, axis=1), -1)
    n14 = units.index("N14")
    classes = []
    for k in range(values.shape[0]):
        rows = np.flatnonzero(owner == k)
        fixed = int(np.sum(d[rows, n14] != 0))
        value = values[k, n14] / joint.SCALE
        cls = {6: "capacity", 10: "counting"}.get(value) if fixed >= RELIABLE else None
        classes.append(cls)
    return classes


def loo_labels(rows, units, tag):
    d_all = joint.matrix([(s, t) for s, t, _ in rows], units)
    base = units.index("N01")
    labels = []
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        d = d_all[keep]
        _, values = joint.search(d, base, K, RESTARTS, rng_for("pe-loo", tag, i))
        classes = system_classes(d, values, units)
        balanced = d_all[i] @ values.T == 0
        hit = {c for c, b in zip(classes, balanced) if b and c}
        labels.append(hit.pop() if len(hit) == 1 else None)
    return labels


def fisher(rows, labels, signs, group, towards):
    other = "counting" if towards == "capacity" else "capacity"
    has = [bool(group & signs.get(r[2], set())) for r in rows]
    a = sum(1 for l, h in zip(labels, has) if l == towards and h)
    b = sum(1 for l, h in zip(labels, has) if l == towards and not h)
    c = sum(1 for l, h in zip(labels, has) if l == other and h)
    d = sum(1 for l, h in zip(labels, has) if l == other and not h)
    return {"group": sorted(group), "towards": towards, "with_group": {towards: a, other: c},
            "tablets": {towards: a + b, other: c + d}, "p": S.fisher_greater(a, b, c, d)}


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    rows, units, _ = informative(N.tablets("uruk-iii") + N.tablets("uruk-iv"))
    labels = loo_labels(rows, units, "control")
    signs = sign_index(["uruk-iii", "uruk-iv"])
    gate = fisher(rows, labels, signs, {"SZE"}, "capacity")
    c1 = fisher(rows, labels, signs, {"UDU", "SAL"}, "counting")
    print("control labels", collections.Counter(labels), "gate", gate, "C1", c1, flush=True)
    result = {"control": {"labels": dict(collections.Counter(str(l) for l in labels)),
                          "gate_test": gate, "C1": c1,
                          "tablets": [{"designation": r[2], "label": l} for r, l in zip(rows, labels)]},
              "gate": gate["p"] < 0.05}
    if result["gate"]:
        rows, units, _ = informative(N.tablets("proto-elamite"))
        labels = loo_labels(rows, units, "target")
        signs = sign_index(["proto-elamite"])
        t1 = fisher(rows, labels, signs, GRAIN, "capacity")
        t2 = fisher(rows, labels, signs, BEINGS, "counting")
        print("target labels", collections.Counter(labels), "t1", t1, "t2", t2, flush=True)
        result["target"] = {"labels": dict(collections.Counter(str(l) for l in labels)),
                            "test_grain": {**t1, "passed": t1["p"] < 0.025},
                            "test_beings": {**t2, "passed": t2["p"] < 0.025},
                            "tablets": [{"designation": r[2], "label": l,
                                         "signs": sorted(signs.get(r[2], set()))}
                                        for r, l in zip(rows, labels)]}
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
