"""Letter-level unknown-word model for the lexicon segmenter (word segmentation v3)."""
from collections import Counter
import hashlib
import heapq
import json
import math

from voynich.decipher import ALPHABET
from voynich.segmentation import Segmenter

START, END = '^', '$'
ELISIONS = ('l', 'd', 's', 'c', 'n', 'm', 't', 'v', 'dell', 'all', 'nell', 'sull', 'dall',
            'coll', 'quell', 'quest', 'bell', 'sant')


class Spelling:
    """Interpolated Witten-Bell letter n-gram over word types, with an end symbol."""

    def __init__(self, words, order):
        if order < 1: raise ValueError('Order must be positive')
        self.order = order
        self.counts = {}
        for word in sorted(set(words)):
            if not word or any(c not in ALPHABET for c in word): raise ValueError('Bad word form: ' + repr(word))
            padded = START * (order - 1) + word + END
            for i in range(order - 1, len(padded)):
                for k in range(order):
                    self.counts.setdefault(padded[i - k:i], Counter())[padded[i]] += 1
        self.totals = {h: (sum(c.values()), len(c)) for h, c in self.counts.items()}
        self.uniform = 1 / (len(ALPHABET) + 1)
        self.cache = {}

    def probability(self, history, char):
        key = (history, char)
        if key not in self.cache:
            lower = self.probability(history[1:], char) if history else self.uniform
            total, types = self.totals.get(history, (0, 0))
            self.cache[key] = lower if not total else (self.counts[history][char] + types * lower) / (total + types)
        return self.cache[key]

    def cost(self, word):
        padded = START * (self.order - 1) + word + END
        return -sum(math.log(self.probability(padded[i - self.order + 1:i], padded[i]))
                    for i in range(self.order - 1, len(padded)))

    def digest(self):
        raw = json.dumps(dict(order=self.order, counts={h: dict(sorted(c.items())) for h, c in sorted(self.counts.items())}),
                         separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()


def unknown_rate(tokens, fixed_forms):
    """Good-Turing share of training tokens that are singletons outside a fixed form list."""
    counts = Counter(tokens)
    return sum(1 for w, n in counts.items() if n == 1 and w not in fixed_forms) / sum(counts.values())


def join_elisions(segmented):
    words = segmented.split()
    out = []
    for word in words:
        if out and out[-1] in ELISIONS and word[0] in 'aeiouh':
            out[-1] += word
        else:
            out.append(word)
    return ' '.join(out)


class SpellingSegmenter(Segmenter):
    """The frozen segmenter with the flat unknown cost replaced by -log p_unk + spelling cost."""

    def __init__(self, model, spelling, rate, elision=False, **parameters):
        super().__init__(model, **parameters)
        if not 0 < rate < 1: raise ValueError('Unknown rate must be in (0, 1)')
        self.spelling, self.rate_cost, self.elision = spelling, -math.log(rate), elision

    def unknown_costs(self, text, start):
        """Spelling costs of every unknown candidate starting at start, built incrementally."""
        order = self.spelling.order
        history = START * (order - 1)
        running, costs = 0., {}
        for end in range(start + 1, min(len(text), start + self.maximum) + 1):
            char = text[end - 1]
            running -= math.log(self.spelling.probability(history[len(history) - order + 1:] if order > 1 else '', char))
            history += char
            tail = history[len(history) - order + 1:] if order > 1 else ''
            costs[end] = self.rate_cost + running - math.log(self.spelling.probability(tail, END))
        return costs

    def segment(self, text):
        if any(c.isspace() for c in text):
            raise ValueError("Segmenter expects a string without supplied boundaries")
        states = [{} for _ in range(len(text) + 1)]
        states[0]["<s>"] = (0., None)
        for start in range(len(text)):
            active = heapq.nsmallest(self.beam if self.bigram else 1, states[start].items(), key=lambda x: (x[1][0], x[0]))
            states[start] = dict(active)
            unknown = self.unknown_costs(text, start)
            for end, word, probability in self.candidates(text, start):
                for previous, (cost, _) in active:
                    if probability is None:
                        extra = unknown[end]
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
        result = " ".join(reversed(words))
        return join_elisions(result) if self.elision else result
