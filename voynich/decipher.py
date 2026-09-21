"""Blind substitution recovery from an unrelated Italian character model."""

from collections import Counter
import math
import random
import unicodedata

import numpy as np

ALPHABET = "abcdefghilmnopqrstuvxyz"  # Naibbe's supported letters; j/k/w normalize to i/c/uu.


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("j", "i").replace("k", "c").replace("w", "uu")
    words = ["".join(c for c in word if c in ALPHABET) for word in text.split()]
    return " ".join(w for w in words if w)


def encode(text, alphabet):
    lookup = {c: i for i, c in enumerate(alphabet)}
    return np.array([lookup[c] for c in text], dtype=np.int32)


def gram_ids(values, base):
    return ((values[:-3] * base + values[1:-2]) * base + values[2:-1]) * base + values[3:]


def language_model(texts, spaces=True):
    alphabet = ALPHABET + (" " if spaces else "")
    base = len(alphabet)
    counts = np.zeros(base**4, dtype=np.float64)
    letters = np.zeros(len(ALPHABET), dtype=np.float64)
    for text in texts:
        text = normalize(text)
        if not spaces:
            text = text.replace(" ", "")
        values = encode(text, alphabet)
        if len(values) >= 4:
            counts += np.bincount(gram_ids(values, base), minlength=base**4)
        letters += np.bincount(values[values < len(ALPHABET)], minlength=len(ALPHABET))
    probabilities = (counts + .05) / (counts.sum() + .05 * len(counts))
    return np.log(probabilities), letters


def frequency_key(values, letter_counts):
    size = len(ALPHABET)
    order_cipher = np.argsort(-np.bincount(values[values < size], minlength=size), kind="stable")
    order_plain = np.argsort(-letter_counts, kind="stable")
    key = np.arange(size + 1, dtype=np.int32)
    key[order_cipher] = order_plain
    return key


def recover(ciphertext, log_probabilities, letter_counts, *, spaces=True, seed=42,
            restarts=6, steps=12000):
    """No plaintext, true key, or paired examples enter this function."""
    alphabet = ALPHABET + (" " if spaces else "")
    values = encode(ciphertext, alphabet)
    base = len(alphabet)
    if len(values) < 4:
        raise ValueError("Need at least four cipher symbols")
    initial = frequency_key(values, letter_counts)
    observed = np.unique(values[values < len(ALPHABET)]).tolist()
    rng = random.Random(seed)

    def score(key):
        return float(log_probabilities[gram_ids(key[values], base)].sum())

    best_key, best_score = initial.copy(), score(initial)
    for restart in range(restarts):
        key = initial.copy() if restart == 0 else best_key.copy()
        if restart:
            for _ in range(5 + restart):
                a, b = rng.sample(range(len(ALPHABET)), 2)
                key[a], key[b] = key[b], key[a]
        current = score(key)
        for step in range(steps):
            a = rng.choice(observed)
            b = rng.randrange(len(ALPHABET))
            if a == b:
                continue
            key[a], key[b] = key[b], key[a]
            candidate = score(key)
            temperature = 12 * (.02 ** (step / steps))
            gain = candidate - current
            if gain >= 0 or rng.random() < math.exp(max(-700, gain / temperature)):
                current = candidate
                if current > best_score:
                    best_key, best_score = key.copy(), current
            else:
                key[a], key[b] = key[b], key[a]
    def decode(key):
        return "".join(alphabet[i] for i in key[values])
    return dict(recovered=decode(best_key), frequency_baseline=decode(initial),
                key=best_key[:len(ALPHABET)].tolist(), language_score=best_score,
                restarts=restarts, steps_per_restart=steps)


def restore_spaces(text, word_counts, maximum=24):
    """Dictionary-only segmentation; a separate, explicitly imperfect recovery step."""
    total = sum(word_counts.values())
    costs = {w: -math.log(n / total) for w, n in word_counts.items()}
    best = [0.] + [math.inf] * len(text)
    previous = [0] * (len(text) + 1)
    for end in range(1, len(text) + 1):
        for start in range(max(0, end - maximum), end):
            word = text[start:end]
            candidate = best[start] + costs.get(word, 15 + .5 * len(word))
            if candidate < best[end]:
                best[end], previous[end] = candidate, start
    words, end = [], len(text)
    while end:
        start = previous[end]
        words.append(text[start:end]); end = start
    return " ".join(reversed(words))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]
