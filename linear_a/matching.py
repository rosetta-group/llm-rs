"""Lexical matching of syllabic words against spelled lexicons, with a phonotactic null.

Distance between two syllable strings is a weighted edit distance: a syllable substitution costs
0 when consonant and vowel agree, 0.5 when one of them differs, 1 when both differ; insertion and
deletion cost 1. An unknown vowel ``?`` agrees with every vowel. The distance is divided by the
longer length. A word *matches* language L when its nearest L form is within ``theta``.

The null keeps a sample's phonotactics and destroys its lexical identity: pseudo-words drawn from
a syllable bigram model fitted to the sample itself, with the sample's length distribution.
"""

import numpy as np
from numba import njit, prange

from linear_a.spelling import CONSONANTS, VOWELS

_C = {"": 0, **{c: i + 1 for i, c in enumerate(CONSONANTS)}}
_V = {**{v: i for i, v in enumerate(VOWELS)}, "?": len(VOWELS)}
UNKNOWN_VOWEL = len(VOWELS)


def encode(words):
    """Flatten syllable tuples into consonant codes, vowel codes, offsets and lengths."""
    lengths = np.array([len(w) for w in words], dtype=np.int64)
    offsets = np.zeros(len(words), dtype=np.int64)
    if len(words):
        offsets[1:] = np.cumsum(lengths)[:-1]
    cons = np.array([_C[c] for w in words for c, _ in w], dtype=np.int8)
    vows = np.array([_V[v] for w in words for _, v in w], dtype=np.int8)
    return cons, vows, offsets, lengths


@njit(cache=True)
def _distance(ac, av, a0, la, bc, bv, b0, lb, row, prev):
    for j in range(lb + 1):
        prev[j] = j
    for i in range(1, la + 1):
        row[0] = i
        ci = ac[a0 + i - 1]
        vi = av[a0 + i - 1]
        for j in range(1, lb + 1):
            cost = 0.0
            if bc[b0 + j - 1] != ci:
                cost += 0.5
            vj = bv[b0 + j - 1]
            if vj != vi and vj != UNKNOWN_VOWEL and vi != UNKNOWN_VOWEL:
                cost += 0.5
            best = prev[j - 1] + cost
            if prev[j] + 1.0 < best:
                best = prev[j] + 1.0
            if row[j - 1] + 1.0 < best:
                best = row[j - 1] + 1.0
            row[j] = best
        for j in range(lb + 1):
            prev[j] = row[j]
    return prev[lb]


@njit(parallel=True, cache=True)
def _nearest(wc, wv, wo, wl, lc, lv, lo, ll):
    n = wl.shape[0]
    best_d = np.full(n, 10.0)
    best_i = np.full(n, -1, dtype=np.int64)
    for k in prange(n):
        la = wl[k]
        row = np.zeros(64)
        prev = np.zeros(64)
        bd = 10.0
        bi = -1
        for m in range(ll.shape[0]):
            lb = ll[m]
            if lb > 60 or la > 60:
                continue
            longest = la if la > lb else lb
            gap = la - lb if la > lb else lb - la
            if gap / longest >= bd:
                continue
            d = _distance(wc, wv, wo[k], la, lc, lv, lo[m], lb, row, prev) / longest
            if d < bd:
                bd = d
                bi = m
                if d == 0.0:
                    break
        best_d[k] = bd
        best_i[k] = bi
    return best_d, best_i


class Matcher:
    """Nearest-form distances to each language, cached per word."""

    def __init__(self, lexicons):
        self.names = list(lexicons)
        self.forms = {name: list(lexicons[name]) for name in self.names}
        self.headwords = lexicons
        self.encoded = {name: encode(self.forms[name]) for name in self.names}
        self.cache = {name: {} for name in self.names}

    def nearest(self, words, name):
        cache = self.cache[name]
        missing = sorted({w for w in words if w not in cache})
        if missing:
            d, i = _nearest(*encode(missing), *self.encoded[name])
            for word, dist, index in zip(missing, d, i):
                cache[word] = (float(dist), int(index))
        return [cache[w] for w in words]

    def distances(self, words, name):
        return np.array([d for d, _ in self.nearest(words, name)])

    def best_form(self, word, name):
        dist, index = self.nearest([word], name)[0]
        form = self.forms[name][index]
        return dist, form, self.headwords[name][form]


class BigramNull:
    """Syllable bigram model with start and end states, sampled at a fixed length."""

    def __init__(self, words, rng):
        self.rng = rng
        self.next = {}
        for word in words:
            states = ("<s>",) + tuple(word) + ("</s>",)
            for a, b in zip(states, states[1:]):
                self.next.setdefault(a, []).append(b)

    def sample(self, length, tries=500):
        for _ in range(tries):
            state, word = "<s>", []
            while len(word) <= length:
                options = self.next[state]
                state = options[self.rng.integers(len(options))]
                if state == "</s>":
                    break
                word.append(state)
            if len(word) == length and state == "</s>":
                return tuple(word)
        # Fallback keeps the length exactly: forward walk that ignores the end state.
        state, word = "<s>", []
        while len(word) < length:
            options = [o for o in self.next.get(state, []) if o != "</s>"] or self.next["<s>"]
            state = options[self.rng.integers(len(options))]
            if state == "</s>":
                continue
            word.append(state)
        return tuple(word)

    def samples(self, lengths, repeats):
        return [[self.sample(n) for n in lengths] for _ in range(repeats)]


def language_scores(matcher, words, thetas, rng, null_repeats=10, null_words=None):
    """Excess matches over the phonotactic null for every language and threshold.

    Returns ``{theta: {language: {matches, null_mean, null_sd, excess_rate, z}}}``. The null
    standard deviation is floored at 1 so a near-constant null cannot produce a huge z.
    """
    words = list(words)
    if null_words is None:
        null = BigramNull(words, rng)
        null_words = null.samples([len(w) for w in words], null_repeats)
    scores = {theta: {} for theta in thetas}
    for name in matcher.names:
        real = matcher.distances(words, name)
        nulls = [matcher.distances(sample, name) for sample in null_words]
        for theta in thetas:
            count = int((real <= theta).sum())
            null_counts = np.array([(d <= theta).sum() for d in nulls], dtype=float)
            mean, sd = null_counts.mean(), null_counts.std(ddof=1)
            scores[theta][name] = {
                "matches": count,
                "null_mean": float(mean),
                "null_sd": float(sd),
                "excess_rate": float((count - mean) / len(words)),
                "z": float((count - mean) / max(sd, 1.0)),
            }
    return scores


def ranking(scores):
    return sorted(scores, key=lambda name: scores[name]["z"], reverse=True)
