"""Linear A round three: entry words against proper-name lexicons, validated on DĀMOS.

    python -m experiments.linear_a_names sources | develop | freeze | verify | test | linear-a

See experiments/linear-a-names/PROTOCOL.md.
"""

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiments.linear_a_context import damos_items, split_types
from linear_a import contexts, controls, corpus, names
from linear_a.matching import BigramNull, Matcher, language_scores, ranking
from linear_a.spelling import render

OUT = Path("experiments/linear-a-names")
NAMES = names.NAMES_DIR
THETAS = (0.0, 0.2)
MIN_LENGTHS = (2, 3)
DRAWS = 20
NULL_REPEATS = 10
LINEAR_A_NULL_REPEATS = 50
Z_MIN = 3.0
POWER = 0.9
FALSE_MAX = 0.05
SOURCE_FILES = [
    "laman_names.csv", "aemw/ugarit/gloss-qpn.json", "aemw/amarna/gloss-qpn.json",
    "aemw/alalakh/idrimi/gloss-qpn.json", "rimanum/gloss-qpn.json",
    "aemw/amarna/metadata.json", "aemw/ugarit/metadata.json", "rimanum/metadata.json",
]
FROZEN = [
    "linear_a/spelling.py", "linear_a/lexicons.py", "linear_a/names.py", "linear_a/matching.py",
    "linear_a/contexts.py", "linear_a/controls.py", "linear_a/corpus.py",
    "experiments/linear_a_context.py", "experiments/linear_a_names.py",
    "experiments/linear-a-names/PROTOCOL.md", "experiments/linear-a-names/sources.json",
    "experiments/linear-a-names/development-results.json",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(_):
    record = {
        "retrieved": "2026-09-23",
        "files": [{"path": str(NAMES / f), "sha256": sha256(NAMES / f)} for f in SOURCE_FILES],
        "origins": {
            "laman_names.csv": {"url": "https://laman.hittites.org/export/search/?",
                                "license": "CC BY-SA 4.0 (Akman, Cammarosano, Kryszeń)"},
            "aemw-*.zip, rimanum.zip": {"url": "https://oracc.museum.upenn.edu/json/",
                                        "license": "CC0 (Oracc metadata.json)"},
            "Greek names": {"source": "../linear-a/sources.json (Wiktionary Ancient Greek)"},
            "DĀMOS": {"source": "../linear-a-context/sources.json"},
        },
    }
    (OUT / "sources.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(len(record["files"]), "files hashed")


def entry_words(types, min_length):
    return sorted(w for w, label in types.items() if label == "entry" and len(w) >= min_length)


def linear_a_types():
    return contexts.type_labels((w, l) for _, w, l in contexts.linear_a_tokens(corpus.load()))


def matcher_for(min_length):
    return Matcher(names.build_name_lexicons(min_length))


def run_draws(matcher, pool, size, theta, tag):
    outcomes, details = [], []
    for d in range(DRAWS):
        rng = controls.rng_for(tag, theta, size, d)
        pick = [pool[i] for i in rng.choice(len(pool), size=size, replace=False)]
        null = BigramNull(pick, rng)
        scores = language_scores(matcher, pick, [theta], rng, null_words=null.samples(
            [len(w) for w in pick], NULL_REPEATS))[theta]
        outcomes.append(controls.identified(scores, Z_MIN))
        details.append({n: round(scores[n]["z"], 2) for n in ranking(scores)})
    greek_z = sorted(d["Greek"] for d in details)
    return {"greek_identified": outcomes.count("Greek") / DRAWS,
            "other_identified": sum(1 for o in outcomes if o not in (None, "Greek")) / DRAWS,
            "greek_z_median": greek_z[len(greek_z) // 2],
            "winners": dict(collections.Counter(o or "none" for o in outcomes)), "z": details}


def develop(_):
    la = linear_a_types()
    mainland = split_types(damos_items()["mainland"])
    cells = {}
    for min_length in MIN_LENGTHS:
        matcher = matcher_for(min_length)
        pool = entry_words(mainland, min_length)
        size = len(entry_words(la, min_length))
        for theta in THETAS:
            cell = run_draws(matcher, pool, size, theta, f"names-dev-{min_length}")
            cells[(theta, min_length)] = {"pool": len(pool), "size": size, **cell}
            print(theta, min_length, size, len(pool), cell["greek_identified"],
                  cell["other_identified"], cell["winners"], cell["greek_z_median"], flush=True)
    order = {(t, m): i for i, (t, m) in enumerate([(t, m) for t in THETAS for m in MIN_LENGTHS])}
    eligible = [k for k in cells if cells[k]["other_identified"] <= FALSE_MAX]
    chosen = max(eligible, key=lambda k: (cells[k]["greek_identified"], -order[k])) if eligible else None
    results = {
        "lexicon_sizes": {m: {n: len(f) for n, f in matcher_for(m).forms.items()} for m in MIN_LENGTHS},
        "cells": [{"theta": t, "min_length": m, **v} for (t, m), v in cells.items()],
        "chosen": {"theta": chosen[0], "min_length": chosen[1]} if chosen else None,
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
    src = json.loads((OUT / "sources.json").read_text())
    bad = [p for p, h in record["files"].items() if sha256(p) != h]
    bad += [f["path"] for f in src["files"] if sha256(f["path"]) != f["sha256"]]
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
    theta, min_length = record["theta"], record["min_length"]
    knossos = split_types(damos_items()["knossos"])
    pool = entry_words(knossos, min_length)
    size = len(entry_words(linear_a_types(), min_length))
    cell = run_draws(matcher_for(min_length), pool, size, theta, "names-test")
    gate = cell["greek_identified"] >= POWER and cell["other_identified"] <= FALSE_MAX
    results = {"theta": theta, "min_length": min_length, "pool": len(pool), "size": size, **cell,
               "gate": {"passed": gate, "power": POWER, "false_max": FALSE_MAX}}
    (OUT / "test-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(cell["greek_identified"], cell["other_identified"], cell["winners"],
          cell["greek_z_median"], "gate passed" if gate else "gate FAILED")


def run_linear_a(_):
    record = verify()
    if not json.loads((OUT / "test-results.json").read_text())["gate"]["passed"]:
        sys.exit("Knossos gate failed; the protocol does not run Linear A")
    if (OUT / "linear-a-results.json").exists():
        sys.exit("linear-a-results.json exists; refusing to overwrite")
    theta, min_length = record["theta"], record["min_length"]
    matcher = matcher_for(min_length)
    words = entry_words(linear_a_types(), min_length)
    rng = controls.rng_for("names-linear-a")
    scores = language_scores(matcher, words, [theta], rng, null_repeats=LINEAR_A_NULL_REPEATS)[theta]
    top = controls.identified(scores, Z_MIN)
    matches = {}
    for language in matcher.names:
        rows = []
        for w in words:
            d, form, heads = matcher.best_form(w, language)
            if d <= theta:
                rows.append({"linear_a": render(w), "name": render(form), "headwords": heads[:3]})
        matches[language] = rows
    results = {"theta": theta, "min_length": min_length, "words": len(words), "scores": scores,
               "ranking": ranking(scores), "identified": top, "matches": matches}
    (OUT / "linear-a-results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print({n: round(scores[n]["z"], 2) for n in ranking(scores)}, "identified:", top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sources", "develop", "freeze", "verify", "test", "linear-a"])
    stage = parser.parse_args().stage
    {"sources": sources, "develop": develop, "freeze": freeze, "verify": verify, "test": test,
     "linear-a": run_linear_a}[stage](None)


if __name__ == "__main__":
    main()
