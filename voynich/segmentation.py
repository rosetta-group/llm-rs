"""Lexicon segmentation with word-transition costs and explicit unknown forms."""

from collections import Counter
import heapq
import math

from voynich.decipher import normalize


def fit(texts, lexicon):
    counts, pairs = Counter(), Counter()
    for text in texts:
        words = normalize(text).split()
        counts.update(words)
        pairs.update(zip(["<s>"] + words, words))
    following = {}
    for (a, b), n in pairs.items():
        following.setdefault(a, {})[b] = n
    return dict(counts=dict(counts), following=following,
                lexicon=sorted(set(lexicon) | set(counts)))


class Segmenter:
    def __init__(self, model, alpha=.1, bigram=.5, unknown=3., beam=8, maximum=32):
        self.counts = model["counts"]
        self.following = model["following"]
        self.totals = {w: sum(c.values()) for w, c in self.following.items()}
        self.alpha, self.bigram, self.unknown = alpha, bigram, unknown
        self.beam, self.maximum = beam, maximum
        self.total = sum(self.counts.values()) + alpha * len(model["lexicon"])
        self.trie = {}
        for word in model["lexicon"]:
            node = self.trie
            for char in word:
                node = node.setdefault(char, {})
            node[""] = True

    def candidates(self, text, start):
        node, known = self.trie, set()
        for end in range(start + 1, min(len(text), start + self.maximum) + 1):
            node = node.get(text[end - 1])
            if node is None:
                break
            if "" in node:
                known.add(end)
        for end in range(start + 1, min(len(text), start + self.maximum) + 1):
            word = text[start:end]
            probability = (self.counts.get(word, 0) + self.alpha) / self.total if end in known else None
            yield end, word, probability

    def segment(self, text):
        if any(c.isspace() for c in text):
            raise ValueError("Segmenter expects a string without supplied boundaries")
        # State is previous word; retain the cheapest prefix for each state.
        states = [{} for _ in range(len(text) + 1)]
        states[0]["<s>"] = (0., None)
        for start in range(len(text)):
            active = heapq.nsmallest(self.beam if self.bigram else 1, states[start].items(), key=lambda x: (x[1][0], x[0]))
            states[start] = dict(active)
            for end, word, probability in self.candidates(text, start):
                for previous, (cost, _) in active:
                    if probability is None:
                        extra = 15 + self.unknown * len(word)
                    else:
                        transition = (self.following.get(previous, {}).get(word, 0) + 20 * probability) / (self.totals.get(previous, 0) + 20)
                        extra = -math.log((1 - self.bigram) * probability + self.bigram * transition)
                    candidate = cost + extra
                    if candidate < states[end].get(word, (math.inf,))[0]:
                        states[end][word] = (candidate, (start, previous))
        if not text:
            return ""
        word = min(states[-1], key=lambda w: (states[-1][w][0], w))
        end, words = len(text), []
        while end:
            words.append(word)
            end, word = states[end][word][1]
        return " ".join(reversed(words))


def boundaries(text):
    positions, offset = set(), 0
    for word in text.split()[:-1]:
        offset += len(word)
        positions.add(offset)
    return positions
