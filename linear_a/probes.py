"""Algorithms for the round-four probes (see experiments/linear-a-probes/PROTOCOL.md)."""

import collections
import math
import zlib

import numpy as np

from linear_a.matching import BigramNull
from linear_a.spelling import VOWELS, render

# ---- shared ---------------------------------------------------------------------------------


def null_corpora(words, rng, repeats):
    """Pseudo-word sets from a syllable bigram model of ``words``, same lengths."""
    null = BigramNull(list(words), rng)
    return [[null.sample(len(w)) for w in words] for _ in range(repeats)]


def upper_p(observed, null_values):
    """One-sided p with the +1 correction."""
    null_values = np.asarray(null_values, dtype=float)
    return float(((null_values >= observed).sum() + 1) / (len(null_values) + 1))


def half(word):
    return zlib.crc32(render(word).encode()) % 2


# ---- probe 1: consonant skeletons -----------------------------------------------------------

_EQUIVALENT = {"d": "t", "z": "s"}
_DROPPABLE = set("smnr")


def word_skeleton(word):
    """Consonants of a syllable word, glide j dropped, d written t."""
    return "".join(_EQUIVALENT.get(c, c) for c, _ in word if c and c != "j")


def skeleton_matches(egyptian, skeleton):
    """True if ``skeleton`` is ``egyptian`` with some s, m, n or r deleted (t ~ d)."""
    egyptian = "".join(_EQUIVALENT.get(c, c) for c in egyptian)

    def walk(i, j):
        if j == len(skeleton):
            return all(c in _DROPPABLE for c in egyptian[i:])
        if i == len(egyptian):
            return False
        if egyptian[i] == skeleton[j] and walk(i + 1, j + 1):
            return True
        return egyptian[i] in _DROPPABLE and walk(i + 1, j)

    return bool(skeleton) and walk(0, 0)


# ---- probe 4: stems -------------------------------------------------------------------------


def stem(word, syllables=2):
    """First ``syllables`` syllables with the vowel of the last one free."""
    if word is None or len(word) < syllables:
        return None
    head = tuple(word[: syllables - 1])
    return head + ((word[syllables - 1][0], "?"),)


def starts_with(word, prefix):
    if len(word) < len(prefix):
        return False
    for (c, v), (pc, pv) in zip(word, prefix):
        if c != pc or (pv != "?" and v != pv):
            return False
    return True


# ---- probe 5: profiles ----------------------------------------------------------------------

FEATURES = ("suffix_alternation", "prefix_alternation", *[f"final_{v}" for v in VOWELS],
            "final_entropy", "mean_length", "vowel_initial")


def profile(words):
    words = [w for w in words if len(w) >= 2]
    by_head = collections.Counter(w[:-1] for w in words)
    by_tail = collections.Counter(w[1:] for w in words)
    finals = collections.Counter(w[-1] for w in words)
    total = len(words)
    entropy = -sum(n / total * math.log2(n / total) for n in finals.values())
    vowels = collections.Counter(w[-1][1] for w in words)
    return np.array([
        sum(by_head[w[:-1]] > 1 for w in words) / total,
        sum(by_tail[w[1:]] > 1 for w in words) / total,
        *[vowels[v] / total for v in VOWELS],
        entropy,
        float(np.mean([len(w) for w in words])),
        sum(w[0][0] == "" for w in words) / total,
    ])


def nearest_profiles(target_samples, reference_samples):
    """For each target sample, the nearest reference on features standardised over all samples."""
    everything = np.vstack([*target_samples, *[p for ps in reference_samples.values() for p in ps]])
    scale = everything.std(axis=0)
    scale[scale == 0] = 1.0
    centres = {name: np.mean(ps, axis=0) for name, ps in reference_samples.items()}
    out = []
    for p in target_samples:
        distance = {name: float(np.linalg.norm((p - c) / scale)) for name, c in centres.items()}
        out.append(min(distance, key=distance.get))
    return out, {name: c.tolist() for name, c in centres.items()}


# ---- probe 6: substitutions -----------------------------------------------------------------


def _codes(words):
    return np.array([[hash(syllable) for syllable in w] for w in words], dtype=np.int64)


def one_substitutions(words, targets_by_length, min_length=3):
    """``Counter((position, from, to))`` and the pairs, for equal-length one-syllable changes."""
    counts, pairs = collections.Counter(), []
    by_length = collections.defaultdict(list)
    for w in words:
        if len(w) >= min_length:
            by_length[len(w)].append(w)
    for n, group in by_length.items():
        targets = targets_by_length.get(n)
        if not targets:
            continue
        a, b = _codes(group), _codes(targets)
        differ = a[:, None, :] != b[None, :, :]
        rows, cols = np.nonzero(differ.sum(axis=2) == 1)
        for r, c in zip(rows, cols):
            i = int(np.argmax(differ[r, c]))
            w, v = group[r], targets[c]
            position = "final" if i == n - 1 else "initial" if i == 0 else "medial"
            key = (position, w[i], v[i])
            counts[key] += 1
            pairs.append((w, v, key))
    return counts, pairs


def rule_key(key):
    position, (c1, v1), (c2, v2) = key
    return f"{position}:{c1}{v1}->{c2}{v2}"
