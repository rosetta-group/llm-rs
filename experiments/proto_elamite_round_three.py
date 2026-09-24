"""Proto-Elamite round three: object signs by arithmetic system label, control first.

    python -m experiments.proto_elamite_round_three

See experiments/proto-elamite-signs/PROTOCOL.md. Writes results.json and refuses to overwrite it.
"""

import collections
import json
import sys
from pathlib import Path

from proto_elamite import numerals as N
from proto_elamite import signs as S

OUT = Path("experiments/proto-elamite-signs")
GRAIN = {"M288", "M036", "M297"}
BEINGS = {"M388", "M124", "M346", "M367", "M006", "M362", "M376"}


def labelled(folders):
    rows = []
    for folder in folders:
        for record in N.records(folder):
            if not N.is_administrative(record):
                continue
            atf = (record.get("inscription") or {}).get("atf") or ""
            entries, total, notations, clean = N.parse(atf)
            t = {"entries": entries, "total": total, "clean": clean,
                 "usable": clean and len(entries) >= 2 and total is not None}
            lab = S.label(t)
            if lab:
                rows.append({"designation": record.get("designation"), "label": lab,
                             "signs": sorted(S.tablet_signs(atf))})
    return rows


def test(rows, has, towards):
    other = "counting" if towards == "capacity" else "capacity"
    a = sum(1 for r in rows if r["label"] == towards and has(r))
    b = sum(1 for r in rows if r["label"] == towards and not has(r))
    c = sum(1 for r in rows if r["label"] == other and has(r))
    d = sum(1 for r in rows if r["label"] == other and not has(r))
    return {"towards": towards, "with_group": {towards: a, other: c},
            "tablets": {towards: a + b, other: c + d}, "p": S.fisher_greater(a, b, c, d)}


def table(rows, minimum=3):
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        for s in r["signs"]:
            counts[s][r["label"]] += 1
    return {s: dict(c) for s, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))
            if sum(c.values()) >= minimum}


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    control = labelled(["uruk-iii", "uruk-iv"])
    gate = test(control, lambda r: any(s.startswith("ŠE") for s in r["signs"]), "capacity")
    print("control", collections.Counter(r["label"] for r in control), gate, flush=True)
    result = {"control": {"labels": dict(collections.Counter(r["label"] for r in control)),
                          "gate_test": gate, "signs": table(control)},
              "gate": gate["p"] < 0.05}
    if result["gate"]:
        target = labelled(["proto-elamite"])
        t1 = test(target, lambda r: bool(GRAIN & set(r["signs"])), "capacity")
        t2 = test(target, lambda r: bool(BEINGS & set(r["signs"])), "counting")
        print("target", collections.Counter(r["label"] for r in target), t1, t2, flush=True)
        result["target"] = {"labels": dict(collections.Counter(r["label"] for r in target)),
                            "test_grain": {**t1, "passed": t1["p"] < 0.025},
                            "test_beings": {**t2, "passed": t2["p"] < 0.025},
                            "signs": table(target),
                            "tablets": target}
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
