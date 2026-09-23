"""Linear A round one: freeze, Linear B control, then Linear A only if the gate passes.

    python -m experiments.linear_a_round_one freeze | verify | control | linear-a

See experiments/linear-a/PROTOCOL.md.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from linear_a import anchors, arithmetic, controls, corpus, lexicons
from linear_a.matching import BigramNull, Matcher, language_scores, ranking
from linear_a.spelling import render

OUT = Path("experiments/linear-a")
FROZEN = [
    "linear_a/__init__.py", "linear_a/spelling.py", "linear_a/lexicons.py",
    "linear_a/matching.py", "linear_a/controls.py", "linear_a/corpus.py",
    "linear_a/anchors.py", "linear_a/arithmetic.py",
    "experiments/linear_a_development.py", "experiments/linear_a_round_one.py",
    "experiments/linear-a/PROTOCOL.md", "experiments/linear-a/sources.json",
    "experiments/linear-a/development-results.json",
]
CONTROL_KS = (10, 20, 40, 80, 160, 208)
DRAWS = 20
NULL_DRAWS = 40
NULL_REPEATS = 10
LINEAR_A_NULL_REPEATS = 50
GATE_K = 208
POWER = 0.9
FALSE_MAX = 0.05


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings():
    dev = json.loads((OUT / "development-results.json").read_text())
    return dev, dev["chosen_theta"], dev["settings"]["z_min"], dev["settings"]["min_syllables"]


def freeze(_):
    dev, theta, _, _ = settings()
    if theta is None:
        sys.exit("development chose no threshold; nothing to freeze")
    sources = json.loads((OUT / "sources.json").read_text())
    record = {
        "files": {p: sha256(p) for p in FROZEN},
        "sources": {s["path"]: s["sha256"] for s in sources["sources"]},
        "theta": theta,
    }
    (OUT / "freeze.json").write_text(json.dumps(record, indent=1) + "\n")
    print("wrote freeze.json; commit it before running control")


def verify(_=None):
    record = json.loads((OUT / "freeze.json").read_text())
    bad = [p for p, h in {**record["files"], **record["sources"]}.items() if sha256(p) != h]
    if bad:
        sys.exit(f"frozen files changed: {bad}")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(OUT / "freeze.json")],
                             capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(OUT / "freeze.json")])
    if tracked.returncode or dirty.returncode:
        sys.exit("freeze.json is not committed")
    print("freeze verified")
    return record


def setup():
    _, theta, z_min, min_syllables = settings()
    lx = {name: lexicons.build_lexicon(stem, min_syllables=min_syllables)
          for name, stem in lexicons.CANDIDATES.items()}
    linear_a = list(corpus.readable_types(corpus.load(), min_syllables))
    return Matcher(lx), linear_a, theta, z_min, min_syllables


def _strip(text):
    from linear_a.spelling import strip_diacritics
    return strip_diacritics(text).replace("-", "")


def control(_):
    verify()
    if (OUT / "control-results.json").exists():
        sys.exit("control-results.json exists; refusing to overwrite")
    matcher, linear_a, theta, z_min, min_syllables = setup()
    size = len(linear_a)
    noise = controls.noise_model(linear_a, controls.rng_for("noise"))
    myc = lexicons.mycenaean_words(min_syllables=min_syllables)
    pool = sorted(myc)

    def draw(words, rng):
        null = BigramNull(words, rng)
        return language_scores(matcher, words, [theta], rng, null_words=null.samples(
            [len(w) for w in words], NULL_REPEATS))[theta]

    cells = []
    for k in CONTROL_KS:
        outcomes, greek_z, ranks = [], [], []
        for d in range(DRAWS):
            rng = controls.rng_for("control", k, d)
            words, _ = controls.draw_sample(pool, k, size, noise, rng)
            scores = draw(words, rng)
            outcomes.append(controls.identified(scores, z_min))
            greek_z.append(scores["Greek"]["z"])
            ranks.append(ranking(scores).index("Greek") + 1)
        cell = {"k": k, **controls.summarise(outcomes, "Greek"),
                "greek_z_median": float(np.median(greek_z)),
                "greek_rank_median": float(np.median(ranks)),
                "winners": {n: outcomes.count(n) for n in set(outcomes)}}
        cells.append(cell)
        print(json.dumps(cell), flush=True)

    negative = []
    for d in range(NULL_DRAWS):
        rng = controls.rng_for("control-null", d)
        words, _ = controls.draw_sample([], 0, size, noise, rng)
        negative.append(controls.identified(draw(words, rng), z_min))
    false_rate = controls.summarise(negative, None)["any_winner"]

    # All 218 words alone, no noise: match rate and whether the best Greek form is the cognate.
    rng = controls.rng_for("control-all")
    alone = draw(pool, rng)
    greek = [matcher.best_form(w, "Greek") for w in pool]
    matched = [(w, g) for w, g in zip(pool, greek) if g[0] <= theta]
    with_cognate = [(w, g) for w, g in matched if myc[w]["cognate"]]
    right = [w for w, g in with_cognate
             if any(_strip(h) == _strip(myc[w]["cognate"]) for h in g[2])]

    passing = [c["k"] for c in cells if c["k"] <= GATE_K and c["correct"] >= POWER]
    gate = bool(passing) and false_rate <= FALSE_MAX
    results = {
        "theta": theta, "z_min": z_min, "sample_size": size, "mycenaean_pool": len(pool),
        "cells": cells, "negative_false_winner_rate": false_rate,
        "negative_winners": {n: negative.count(n) for n in set(negative) if n},
        "all_mycenaean_no_noise": {
            "scores": alone, "greek_matches": len(matched),
            "matched_with_cited_cognate": len(with_cognate),
            "best_form_is_cited_cognate": len(right),
            "examples": [{"linear_b": myc[w]["linear_b"], "greek_form": render(g[1]),
                          "headwords": g[2][:3], "cited": myc[w]["cognate"]}
                         for w, g in with_cognate[:25]],
        },
        "gate": {"passing_k": passing, "false_winner_rate": false_rate, "passed": gate},
    }
    (OUT / "control-results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print("gate passed" if gate else "gate FAILED", passing, false_rate)


def run_linear_a(_):
    verify()
    ctrl = json.loads((OUT / "control-results.json").read_text())
    if not ctrl["gate"]["passed"]:
        sys.exit("Linear B control gate failed; the protocol does not run Linear A")
    if (OUT / "linear-a-results.json").exists():
        sys.exit("linear-a-results.json exists; refusing to overwrite")
    matcher, linear_a, theta, z_min, _ = setup()
    rng = controls.rng_for("linear-a")
    scores = language_scores(matcher, linear_a, [theta], rng,
                             null_repeats=LINEAR_A_NULL_REPEATS)[theta]
    top = controls.identified(scores, z_min)
    examples = []
    if top:
        for w in linear_a:
            d, form, heads = matcher.best_form(w, top)
            if d <= theta:
                examples.append({"linear_a": render(w), "form": render(form), "headwords": heads[:3],
                                 "distance": d})
    results = {"theta": theta, "z_min": z_min, "words": len(linear_a), "scores": scores,
               "ranking": ranking(scores), "identified": top, "examples": examples}
    (OUT / "linear-a-results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({n: round(scores[n]["z"], 2) for n in ranking(scores)}), "identified:", top)


def descriptive(_):
    """Corpus checks that test the data and the sign values, not a language."""
    data = corpus.load()
    words = list(corpus.readable_types(data, 2))
    totals = arithmetic.check_totals(data)
    results = {
        "ku_ro": {"lines": len(totals), "exact": sum(r["difference"] == 0 for r in totals),
                  "within_one": sum(abs(r["difference"]) <= 1 for r in totals), "rows": totals},
        "toponyms": anchors.anchor_test(words, controls.rng_for("anchors")),
    }
    (OUT / "descriptive-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({"ku_ro": {k: v for k, v in results["ku_ro"].items() if k != "rows"},
                      "toponyms": {k: v for k, v in results["toponyms"].items() if k != "hits"}}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "verify", "control", "linear-a", "descriptive"])
    stage = parser.parse_args().stage
    {"freeze": freeze, "verify": verify, "control": control, "linear-a": run_linear_a,
     "descriptive": descriptive}[stage](None)


if __name__ == "__main__":
    main()
