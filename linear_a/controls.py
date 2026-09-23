"""Known-answer samples shaped like the Linear A word list.

A sample has ``size`` distinct words: ``k`` signal words from a known language and the rest
noise words. Noise words are drawn from a syllable bigram model of the readable Linear A words,
so they look like Linear A and carry no lexicon.
"""

import zlib

import numpy as np

from linear_a.matching import BigramNull, ranking
from linear_a.spelling import VOWELS


def mutate(word, rate, rng, consonants):
    """Replace each syllable's consonant or vowel with probability ``rate``; fill unknown vowels."""
    out = []
    for c, v in word:
        if v == "?":
            v = VOWELS[rng.integers(len(VOWELS))]
        if rng.random() < rate:
            if rng.random() < 0.5:
                c = consonants[rng.integers(len(consonants))]
            else:
                v = VOWELS[rng.integers(len(VOWELS))]
        out.append((c, v))
    return tuple(out)


def draw_sample(signal_pool, k, size, noise_model, rng, forbid=()):
    """``k`` distinct signal words plus noise words up to ``size``, all distinct."""
    index = rng.choice(len(signal_pool), size=k, replace=False)
    words = {signal_pool[i] for i in index}
    signal = set(words)
    forbid = set(forbid)
    lengths = noise_model["lengths"]
    guard = 0
    while len(words) < size and guard < 100 * size:
        guard += 1
        word = noise_model["null"].sample(lengths[rng.integers(len(lengths))])
        if word not in words and word not in forbid:
            words.add(word)
    return sorted(words), signal


def noise_model(linear_a_words, rng):
    return {"null": BigramNull(linear_a_words, rng), "lengths": [len(w) for w in linear_a_words]}


def identified(scores, z_min):
    """The top language when it clears ``z_min``, else ``None``."""
    top = ranking(scores)[0]
    return top if scores[top]["z"] >= z_min else None


def summarise(outcomes, truth):
    """Share of draws where ``truth`` was identified, and where any language was."""
    n = len(outcomes)
    return {
        "draws": n,
        "correct": sum(1 for o in outcomes if o == truth) / n if n else None,
        "any_winner": sum(1 for o in outcomes if o is not None) / n if n else None,
    }


def rng_for(*keys):
    """Deterministic generator per cell, independent of run order."""
    seed = np.random.SeedSequence([zlib.crc32(str(k).encode()) for k in keys])
    return np.random.default_rng(seed)
