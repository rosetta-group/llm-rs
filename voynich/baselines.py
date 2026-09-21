"""Local spelling, layout, and copy predictors. No neural downloads."""

import math
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from .evaluate import summarize, target_signature


def layout_keys(text):
    """Only past positions: no line length or distance to the future line end."""
    column = line = 0
    previous = ""
    for char in text:
        yield (min(column // 8, 8), min(line, 4))
        if char == "\n":
            column = 0
            line = 0 if previous == "\n" else line + 1
        else:
            column += 1
        previous = char


class Ngram:
    def __init__(self, order=3, strength=5.0):
        self.order, self.strength = order, strength
        self.counts = [defaultdict(Counter) for _ in range(order + 1)]
        self.vocab = 256  # Fixed IVTFF character space, including unseen symbols.

    def fit(self, documents):
        for doc in documents:
            text = doc["text"]
            for i, char in enumerate(text):
                if char == "?":
                    continue
                if ord(char) >= self.vocab:
                    raise ValueError("Baseline expects normalized IVTFF characters in range 0..255")
                for n in range(min(self.order, i) + 1):
                    self.counts[n][text[i-n:i]][ord(char)] += 1
        self.distribution.cache_clear()
        return self

    @lru_cache(maxsize=8192)
    def distribution(self, context):
        context = context[-self.order:] if self.order else ""
        n = len(context)
        counts = self.counts[n].get(context, {})
        if n == 0:
            values = np.full(self.vocab, .01)
        else:
            values = self.strength * self.distribution(context[1:])
        for char, count in counts.items():
            values[char] += count
        return values / values.sum()


class Layout:
    def __init__(self, base, strength=50):
        self.base, self.strength = base, strength
        self.counts = defaultdict(Counter)

    def fit(self, documents):
        for doc in documents:
            for i, (key, char) in enumerate(zip(layout_keys(doc["text"]), doc["text"])):
                if char != "?":
                    self.counts[(doc["text"][max(0, i-self.base.order):i], key)][ord(char)] += 1
        return self

    def distribution(self, context, key):
        context = context[-self.base.order:]
        values = self.strength * self.base.distribution(context)
        for char, count in self.counts.get((context, key), {}).items():
            values[char] += count
        return values / values.sum()


def copy_distribution(context, vocab=256):
    """Continue recent matching strings, or strings with one changed character."""
    recent = context[-256:]
    values = np.zeros(vocab)
    for width in (2, 3, 4):
        if len(recent) <= width:
            continue
        query = recent[-width:]
        for i in range(len(recent) - width):
            candidate = recent[i:i+width]
            mismatches = sum(a != b for a, b in zip(query, candidate))
            if mismatches <= 1:
                values[ord(recent[i+width])] += (4 if mismatches == 0 else 1) * width
    return values / values.sum() if values.sum() else None


def score(documents, model, name, copy_weight=0):
    rows = []
    for doc in documents:
        nll = correct = tokens = 0
        text = doc["text"]
        for i, (char, key) in enumerate(zip(text, layout_keys(text))):
            if char == "?":
                continue
            context = text[max(0, i-256):i]
            probabilities = (model.distribution(context, key) if isinstance(model, Layout)
                             else model.distribution(context[-model.order:] if model.order else ""))
            if copy_weight:
                copied = copy_distribution(context)
                if copied is not None:
                    probabilities = (1-copy_weight)*probabilities + copy_weight*copied
            nll -= math.log(float(probabilities[ord(char)]))
            correct += int(int(np.argmax(probabilities)) == ord(char))
            tokens += 1
        rows.append({**{k: doc[k] for k in ("page", "folio", "quire", "currier", "section", "hand")},
                     "model": name, "nll": nll, "correct": correct, "tokens": tokens, "units": tokens,
                     "target_sha256": target_signature(text)})
    return rows


def benchmark(documents, split="validation", names=None):
    train = [d for d in documents if d["split"] == "train"]
    evaluation = [d for d in documents if d["split"] == split]
    if not train or not evaluation:
        raise ValueError("Training and evaluation data must be non-empty")
    names = names or ["frequency", "ngram3", "ngram5", "layout", "copy"]
    results = {}
    base = Ngram(3).fit(train)
    for name in names:
        print(f"Scoring {name} on {split} ({len(evaluation)} pages)", flush=True)
        if name == "frequency":
            model = Ngram(0).fit(train)
        elif name == "ngram3" or name == "copy":
            model = base
        elif name == "ngram5":
            model = Ngram(5).fit(train)
        elif name == "layout":
            model = Layout(base).fit(train)
        else:
            raise ValueError(f"Unknown baseline: {name}")
        rows = score(evaluation, model, name, copy_weight=.2 if name == "copy" else 0)
        results[name] = dict(summary=summarize(rows), pages=rows)
    return results
