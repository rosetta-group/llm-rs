"""Linear A development: pick the match threshold on synthetic controls only.

    python -m experiments.linear_a_development

Reads the pinned lexicons and only the *count and length profile* of readable Linear A words
(for the noise model). Does not read the Mycenaean control and does not match Linear A words.
See experiments/linear-a/PROTOCOL.md.
"""

import json
import math
import time
from pathlib import Path

from linear_a import controls, corpus, lexicons
from linear_a.matching import BigramNull, Matcher, language_scores
from linear_a.spelling import CONSONANTS

OUT = Path("experiments/linear-a")
THETAS = (0.0, 0.2, 0.25, 0.34)
KS = (10, 20, 40, 80, 160)
DRAWS = 20
NULL_DRAWS = 40
NULL_REPEATS = 10
RATE = 0.2
Z_MIN = 3.0
MIN_SYLLABLES = 2
K_CAP = 320  # recorded as k* when no tested k reaches the power target
POWER = 0.9


def main():
    started = time.time()
    lx = {name: lexicons.build_lexicon(stem, min_syllables=MIN_SYLLABLES)
          for name, stem in lexicons.CANDIDATES.items()}
    matcher = Matcher(lx)
    linear_a = list(corpus.readable_types(corpus.load(), MIN_SYLLABLES))
    size = len(linear_a)
    noise = controls.noise_model(linear_a, controls.rng_for("noise"))
    consonants = [""] + list(CONSONANTS)

    def draw(words, rng):
        null = BigramNull(words, rng)
        return language_scores(matcher, words, THETAS, rng,
                               null_words=null.samples([len(w) for w in words], NULL_REPEATS))

    cells = {}
    for truth in matcher.names:
        forms = matcher.forms[truth]
        for k in KS:
            outcomes = {theta: [] for theta in THETAS}
            for d in range(DRAWS):
                rng = controls.rng_for("dev", truth, k, d)
                picked = [forms[i] for i in rng.choice(len(forms), size=k, replace=False)]
                signal = sorted({controls.mutate(w, RATE, rng, consonants) for w in picked})
                words, _ = controls.draw_sample(signal, len(signal), size, noise, rng)
                for theta, scores in draw(words, rng).items():
                    outcomes[theta].append(controls.identified(scores, Z_MIN))
            for theta in THETAS:
                cells[(truth, k, theta)] = controls.summarise(outcomes[theta], truth)
            print(f"{truth:9s} k={k:3d} " + " ".join(
                f"t{theta}:{cells[(truth, k, theta)]['correct']:.2f}" for theta in THETAS),
                flush=True)

    negative = {theta: [] for theta in THETAS}
    for d in range(NULL_DRAWS):
        rng = controls.rng_for("dev-null", d)
        words, _ = controls.draw_sample([], 0, size, noise, rng)
        for theta, scores in draw(words, rng).items():
            negative[theta].append(controls.identified(scores, Z_MIN))

    selection = {}
    for theta in THETAS:
        k_star = {}
        for truth in matcher.names:
            passing = [k for k in KS if cells[(truth, k, theta)]["correct"] >= POWER]
            k_star[truth] = passing[0] if passing else K_CAP
        false_rate = controls.summarise(negative[theta], None)["any_winner"]
        selection[theta] = {
            "k_star": k_star,
            "mean_log2_k_star": sum(math.log2(v) for v in k_star.values()) / len(k_star),
            "false_winner_rate": false_rate,
            "false_winners": sorted({o for o in negative[theta] if o}),
        }
    eligible = [t for t in THETAS if selection[t]["false_winner_rate"] <= 0.05]
    chosen = min(eligible, key=lambda t: selection[t]["mean_log2_k_star"]) if eligible else None

    results = {
        "linear_a_readable_types": size,
        "lexicon_sizes": {name: len(forms) for name, forms in matcher.forms.items()},
        "settings": {"thetas": THETAS, "ks": KS, "draws": DRAWS, "null_draws": NULL_DRAWS,
                     "null_repeats": NULL_REPEATS, "mutation_rate": RATE, "z_min": Z_MIN,
                     "power": POWER, "min_syllables": MIN_SYLLABLES},
        "cells": [{"truth": t, "k": k, "theta": th, **v} for (t, k, th), v in cells.items()],
        "selection": {str(t): v for t, v in selection.items()},
        "chosen_theta": chosen,
        "seconds": round(time.time() - started),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "development-results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({str(t): {k: v for k, v in s.items()} for t, s in selection.items()},
                     indent=1))
    print("chosen theta", chosen)


if __name__ == "__main__":
    main()
