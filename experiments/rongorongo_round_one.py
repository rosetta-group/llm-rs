"""Rongorongo round one: known-answer recovery curve for a bigram HMM, Māori model.

    python -m experiments.rongorongo_round_one

See experiments/rongorongo/PROTOCOL.md. Writes results.json and refuses to overwrite it.
A smoke test during development used Grey's Ko nga mahinga syllables 20,000-22,600; the
protocol passages start at 40% and 70% of each held-out text, away from it.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from linear_a.controls import rng_for
from rongorongo import solver
from rongorongo import syllables as S

OUT = Path("experiments/rongorongo")
SOURCES = Path("artifacts/rongorongo-sources/maori")
TRAIN = ["kotepaiperatapua00barl", "kotekawenatahou00yategoog"]
HELD_OUT = {"narrative": "kongamahingaang00greygoog", "song": "kongamoteateame00greygoog"}
SIZES = (1300, 2600, 5200, 8000)
STARTS = (0.4, 0.7)
RESTARTS, ITERATIONS = 20, 200
GATE_SIZE, GATE_ACCURACY = 2600, 0.9


def stream(name):
    return S.syllables(S.words((SOURCES / f"{name}.txt").read_text(errors="ignore")))


def key_for(variant, rng):
    """Syllable -> list of sign numbers; homophones give 20% of syllables a second sign."""
    signs = rng.permutation(1000)[: 2 * len(S.ALPHABET)]
    key = {s: [int(signs[i])] for i, s in enumerate(S.ALPHABET)}
    if variant == "homophones":
        extra = rng.choice(len(S.ALPHABET), size=len(S.ALPHABET) // 5, replace=False)
        for j, i in enumerate(extra):
            key[S.ALPHABET[i]].append(int(signs[len(S.ALPHABET) + j]))
    return key


def encipher(passage, key, rng):
    return [key[s][rng.integers(len(key[s]))] for s in passage]


def main():
    path = OUT / "results.json"
    if path.exists():
        sys.exit("results.json exists; refusing to overwrite")
    train = [s for name in TRAIN for s in stream(name)]
    start, trans = solver.bigram_model(train, S.ALPHABET)
    held = {genre: stream(name) for genre, name in HELD_OUT.items()}
    cells = []
    for size in SIZES:
        for genre, text in held.items():
            for position in STARTS:
                offset = int(position * len(text))
                passage = text[offset: offset + size]
                for variant in ("one-to-one", "homophones"):
                    rng = rng_for("rongorongo", size, genre, position, variant)
                    key = key_for(variant, rng)
                    signs = encipher(passage, key, rng)
                    began = time.time()
                    decoded, ll = solver.decipher(signs, start, trans, RESTARTS, ITERATIONS,
                                                  seed=int(rng.integers(2**31)))
                    accuracy = float(np.mean([S.ALPHABET[d] == s for d, s in zip(decoded, passage)]))
                    cell = {"size": size, "genre": genre, "start": position, "variant": variant,
                            "sign_types": len(set(signs)), "accuracy": accuracy,
                            "log_likelihood": ll, "seconds": round(time.time() - began, 1)}
                    cells.append(cell)
                    print(json.dumps(cell), flush=True)
    gate_cells = [c for c in cells if c["size"] == GATE_SIZE and c["variant"] == "one-to-one"]
    passed = all(c["accuracy"] >= GATE_ACCURACY for c in gate_cells)
    summary = {}
    for size in SIZES:
        for variant in ("one-to-one", "homophones"):
            acc = [c["accuracy"] for c in cells if c["size"] == size and c["variant"] == variant]
            summary[f"{size} {variant}"] = {"mean": float(np.mean(acc)), "min": float(np.min(acc))}
    result = {"train_syllables": len(train), "held_out_syllables": {g: len(t) for g, t in held.items()},
              "cells": cells, "summary": summary,
              "gate": {"size": GATE_SIZE, "accuracy": GATE_ACCURACY, "passed": passed}}
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(summary, indent=1), "gate passed" if passed else "gate FAILED")


if __name__ == "__main__":
    main()
