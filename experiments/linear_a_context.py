"""Linear A round two: context agreement, validated on DĀMOS Linear B.

    python -m experiments.linear_a_context sources | develop | freeze | verify | test | linear-a

See experiments/linear-a-context/PROTOCOL.md.
"""

import argparse
import collections
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from linear_a import contexts, controls, corpus, lexicons, lexicons_v2
from linear_a.context_test import context_scores, name_flags
from linear_a.matching import Matcher
from linear_a.matching import ranking

OUT = Path("experiments/linear-a-context")
DAMOS = Path("artifacts/linear-a-sources/damos")
THETAS = (0.0, 0.2)
RULES = ("any", "majority")
DRAWS = 20
REPEATS = 1000
LINEAR_A_REPEATS = 10000
Z_MIN = 3.0
POWER = 0.9
FALSE_MAX = 0.05
KNOSSOS_CSORT = 10
FROZEN = [
    "linear_a/spelling.py", "linear_a/lexicons.py", "linear_a/lexicons_v2.py",
    "linear_a/matching.py", "linear_a/contexts.py", "linear_a/context_test.py",
    "linear_a/controls.py", "linear_a/corpus.py", "experiments/linear_a_context.py",
    "experiments/linear-a-context/PROTOCOL.md", "experiments/linear-a-context/sources.json",
    "experiments/linear-a-context/development-results.json",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def damos_items():
    tables = json.loads((DAMOS / "filter.json").read_text())["tables"]
    knossos = {t["id"] for t in tables if t["csort"] == KNOSSOS_CSORT}
    items = {"mainland": [], "knossos": []}
    for path in sorted(glob.glob(str(DAMOS / "items" / "*.json")), key=lambda p: int(Path(p).stem)):
        item = json.loads(Path(path).read_text())["item"]
        if item is None:
            continue
        items["knossos" if int(Path(path).stem) in knossos else "mainland"].append(item)
    return items


def sources(_):
    ids = sorted({t["id"] for t in json.loads((DAMOS / "filter.json").read_text())["tables"]})
    present = sorted(int(Path(p).stem) for p in glob.glob(str(DAMOS / "items" / "*.json")))
    digest = hashlib.sha256()
    for i in present:
        digest.update(f"{i}:{sha256(DAMOS / 'items' / f'{i}.json')}\n".encode())
    record = {
        "name": "DĀMOS: Database of Mycenaean at Oslo", "url": "https://damos.hf.uio.no/",
        "endpoint": "https://damos.hf.uio.no/ajaxitem/{id}/", "retrieved": "2026-09-23",
        "license": "CC BY-NC-SA 4.0 (content)", "ids_listed": len(ids), "ids_present": len(present),
        "missing": sorted(set(ids) - set(present)),
        "filter_sha256": sha256(DAMOS / "filter.json"),
        "items_manifest_sha256": digest.hexdigest(),
        "linear_a_and_lexicons": "../linear-a/sources.json",
    }
    (OUT / "sources.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print({k: v for k, v in record.items() if k != "missing"}, "missing", len(record["missing"]))


def setup():
    named = {name: lexicons_v2.build_named_lexicon(stem) for name, stem in lexicons.CANDIDATES.items()}
    matcher = Matcher({name: {f: r["headwords"] for f, r in lx.items()} for name, lx in named.items()})
    linear_a = contexts.type_labels((w, l) for _, w, l in contexts.linear_a_tokens(corpus.load()))
    return named, matcher, linear_a


def split_types(items):
    return contexts.type_labels((w, l) for _, w, l in contexts.linear_b_tokens(items))


def run_draws(matcher, flags, types, size, theta, tag):
    words_all = sorted(types)
    outcomes, greek_z, details = [], [], []
    for d in range(DRAWS):
        rng = controls.rng_for(tag, theta, d)
        pick = [words_all[i] for i in rng.choice(len(words_all), size=size, replace=False)]
        scores = context_scores(matcher, flags, pick, [types[w] for w in pick], theta, rng, REPEATS)
        outcomes.append(controls.identified(scores, Z_MIN))
        greek_z.append(scores["Greek"]["z"])
        details.append({n: round(scores[n]["z"], 2) for n in ranking(scores)})
    other = sum(1 for o in outcomes if o not in (None, "Greek")) / DRAWS
    return {"greek_identified": outcomes.count("Greek") / DRAWS, "other_identified": other,
            "greek_z_median": float(np.median(greek_z)),
            "winners": dict(collections.Counter(o or "none" for o in outcomes)), "z": details}


def develop(_):
    named, matcher, linear_a = setup()
    mainland = split_types(damos_items()["mainland"])
    size = len(linear_a)
    cells = {}
    for theta in THETAS:
        for rule in RULES:
            flags = {n: name_flags(named[n], rule) for n in matcher.names}
            cells[(theta, rule)] = run_draws(matcher, flags, mainland, size, theta, f"dev-{rule}")
            c = cells[(theta, rule)]
            print(theta, rule, c["greek_identified"], c["other_identified"], c["winners"], flush=True)
    eligible = [k for k in cells if cells[k]["other_identified"] <= FALSE_MAX]
    order = {k: i for i, k in enumerate(cells)}
    chosen = max(eligible, key=lambda k: (cells[k]["greek_identified"], -order[k])) if eligible else None
    labels = collections.Counter(mainland.values())
    results = {
        "sample_size": size, "mainland_types": len(mainland),
        "mainland_labels": dict(labels),
        "linear_a_labels": dict(collections.Counter(linear_a.values())),
        "cells": [{"theta": t, "rule": r, **v} for (t, r), v in cells.items()],
        "chosen": {"theta": chosen[0], "rule": chosen[1]} if chosen else None,
    }
    (OUT / "development-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print("chosen", results["chosen"])


def freeze(_):
    dev = json.loads((OUT / "development-results.json").read_text())
    if dev["chosen"] is None:
        sys.exit("development chose no setting")
    record = {"files": {p: sha256(p) for p in FROZEN}, **dev["chosen"]}
    (OUT / "freeze.json").write_text(json.dumps(record, indent=1) + "\n")
    print("wrote freeze.json; commit before test")


def verify(_=None):
    record = json.loads((OUT / "freeze.json").read_text())
    bad = [p for p, h in record["files"].items() if sha256(p) != h]
    src = json.loads((OUT / "sources.json").read_text())
    tables = json.loads((DAMOS / "filter.json").read_text())
    if sha256(DAMOS / "filter.json") != src["filter_sha256"] or not tables:
        bad.append("filter.json")
    if bad:
        sys.exit(f"frozen files changed: {bad}")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(OUT / "freeze.json")])
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(OUT / "freeze.json")],
                             capture_output=True)
    if dirty.returncode or tracked.returncode:
        sys.exit("freeze.json is not committed")
    print("freeze verified")
    return record


def test(_):
    record = verify()
    if (OUT / "test-results.json").exists():
        sys.exit("test-results.json exists; refusing to overwrite")
    named, matcher, linear_a = setup()
    knossos = split_types(damos_items()["knossos"])
    flags = {n: name_flags(named[n], record["rule"]) for n in matcher.names}
    cell = run_draws(matcher, flags, knossos, len(linear_a), record["theta"], "test")
    gate = cell["greek_identified"] >= POWER and cell["other_identified"] <= FALSE_MAX
    results = {"theta": record["theta"], "rule": record["rule"], "knossos_types": len(knossos),
               "knossos_labels": dict(collections.Counter(knossos.values())),
               **cell, "gate": {"passed": gate, "power": POWER, "false_max": FALSE_MAX}}
    (OUT / "test-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(cell["greek_identified"], cell["other_identified"], cell["winners"],
          "gate passed" if gate else "gate FAILED")


def run_linear_a(_):
    record = verify()
    if not json.loads((OUT / "test-results.json").read_text())["gate"]["passed"]:
        sys.exit("Knossos gate failed; the protocol does not run Linear A")
    if (OUT / "linear-a-results.json").exists():
        sys.exit("linear-a-results.json exists; refusing to overwrite")
    named, matcher, linear_a = setup()
    flags = {n: name_flags(named[n], record["rule"]) for n in matcher.names}
    words = sorted(linear_a)
    rng = controls.rng_for("linear-a-context")
    scores = context_scores(matcher, flags, words, [linear_a[w] for w in words], record["theta"],
                            rng, LINEAR_A_REPEATS)
    top = controls.identified(scores, Z_MIN)
    results = {**record, "words": len(words), "scores": scores, "ranking": ranking(scores),
               "identified": top}
    results.pop("files")
    (OUT / "linear-a-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print({n: round(scores[n]["z"], 2) for n in ranking(scores)}, "identified:", top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sources", "develop", "freeze", "verify", "test", "linear-a"])
    stage = parser.parse_args().stage
    {"sources": sources, "develop": develop, "freeze": freeze, "verify": verify, "test": test,
     "linear-a": run_linear_a}[stage](None)


if __name__ == "__main__":
    main()
