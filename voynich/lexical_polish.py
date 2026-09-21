"""Word-level polish of a one-letter unit key.

The character prior decides most letters but tolerates errors that break words. This pass
re-scores single-unit changes with the frozen lexicon segmenter's cost, computed locally in
windows around the unit's occurrences. The character prior shortlists alternatives; the
lexicon cost, in bits, plus the character bits decide. It changes the key only when the
combined cost falls, so it can be audited change by change.
"""
from collections import defaultdict
import heapq
import math
import time

import numpy as np

from voynich.variable_units import _prepare, key_arrays, variable_scores

LN2 = math.log(2)


class LexicalCost:
    """Minimum segmentation cost of a letter string under a fitted Segmenter, in bits."""

    def __init__(self, segmenter):
        self.s = segmenter

    def cost(self, text):
        s = self.s
        if not text:
            return 0.
        states = [{} for _ in range(len(text) + 1)]
        states[0]['<s>'] = 0.
        for start in range(len(text)):
            active = heapq.nsmallest(s.beam if s.bigram else 1, states[start].items(), key=lambda x: (x[1], x[0]))
            states[start] = dict(active)
            for end, word, probability in s.candidates(text, start):
                for previous, cost in active:
                    if probability is None:
                        extra = 15 + s.unknown * len(word)
                    else:
                        transition = (s.following.get(previous, {}).get(word, 0) + 20 * probability) / (s.totals.get(previous, 0) + 20)
                        extra = -math.log((1 - s.bigram) * probability + s.bigram * transition)
                    candidate = cost + extra
                    if candidate < states[end].get(word, math.inf):
                        states[end][word] = candidate
        return min(states[-1].values()) / LN2


def windows(positions, length, radius):
    """Merged [start, end) windows of `radius` letters around each position."""
    spans = []
    for p in sorted(positions):
        start, end = max(0, p - radius), min(length, p + radius + 1)
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return spans


def polish(units, prior, mapping, lexical, weight=1., radius=30, shortlist=5, sweeps=3, cap=1200):
    """Greedy sweeps over units; each accepted change lowers character bits + weight × lexical bits.

    Returns the new mapping, the recovered letters and a record of every accepted change.
    """
    started = time.monotonic()
    inventory, values, freq, costs, offsets, sb, db = _prepare(units, prior)
    base, order = prior.base, prior.order
    first, second = key_arrays(inventory, mapping, prior)
    letters = list(prior.alphabet)
    positions = defaultdict(list)
    for i, u in enumerate(units):
        positions[u].append(i)
    text = list(''.join(mapping[u] for u in units))

    def char_bits(f):
        return variable_scores(f, np.repeat(second, len(f), axis=0), values, freq, costs, offsets, base, order, sb, db)

    current_char = float(char_bits(first)[0])
    changes = []; evaluations = 0
    for sweep in range(sweeps):
        improved = False
        for s in np.argsort(-freq, kind='stable'):
            if time.monotonic() - started > cap:
                break
            unit = inventory[s]; old = int(first[0, s])
            trial = np.repeat(first, base, axis=0); trial[:, s] = np.arange(base)
            bits = char_bits(trial); evaluations += base
            order_ = [k for k in np.argsort(bits, kind='stable') if k != old][:shortlist]
            spans = windows(positions[unit], len(text), radius)
            base_lex = sum(lexical.cost(''.join(text[a:b])) for a, b in spans)
            best, best_total = None, 0.
            for k in order_:
                for p in positions[unit]:
                    text[p] = letters[k]
                lex = sum(lexical.cost(''.join(text[a:b])) for a, b in spans)
                for p in positions[unit]:
                    text[p] = letters[old]
                total = (float(bits[k]) - current_char) + weight * (lex - base_lex)
                if total < best_total - 1e-9:
                    best, best_total, best_lex = int(k), total, lex
            if best is not None:
                for p in positions[unit]:
                    text[p] = letters[best]
                first[0, s] = best; current_char = float(bits[best]); improved = True
                changes.append(dict(sweep=sweep, unit=unit, occurrences=len(positions[unit]), old=letters[old], new=letters[best],
                                    delta_bits=best_total, lexical_delta_bits=best_lex - base_lex))
        if not improved or time.monotonic() - started > cap:
            break
    new_mapping = {u: letters[int(first[0, i])] for i, u in enumerate(inventory)}
    return dict(mapping=new_mapping, recovered=''.join(text), changes=changes, char_bits=current_char,
                evaluations=evaluations, sweeps=sweep + 1, seconds=time.monotonic() - started,
                cap_hit=time.monotonic() - started > cap)
