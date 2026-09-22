"""Rechoose known cipher-piece splits under a fixed key and full character context.

No new pieces or letter mappings are invented. Each token has a whole reading and/or
known prefix+suffix readings. Beam search minimizes character bits plus homophone
choice bits; key/inventory and one whole/split bit per token are constant. The
plaintext length uses the same integer code as CharacterPrior.bits. This is a
conditional code given the fixed role-key, not an unrestricted cipher-family score.
"""
from collections import Counter
import math
import time

from voynich.description_length import integer_bits


def options(token, mapping):
    found = []
    if 'u:' + token in mapping:
        found.append(((token,), ('u:' + token,)))
    for i in range(1, len(token)):
        units = ('p:' + token[:i], 's:' + token[i:])
        if all(u in mapping for u in units):
            found.append(((token[:i], token[i:]), units))
    if not found:
        raise ValueError('No known-key reading for token: ' + token)
    return found


def choice_costs(mapping):
    counts = Counter((u[0], c) for u, c in mapping.items())
    return {u: math.log2(counts[u[0], c]) for u, c in mapping.items()}


def score(segmentation, mapping, prior):
    costs = choice_costs(mapping)
    units = [r + ':' + p for parts in segmentation
             for r, p in zip(('u',) if len(parts) == 1 else ('p', 's'), parts)]
    return prior.bits(''.join(mapping[u] for u in units)) + sum(costs[u] for u in units)


def reparse(tokens, mapping, prior, width=128):
    if width < 1:
        raise ValueError('Beam width must be positive')
    if any(len(c) != 1 or c not in prior.alphabet or u[:2] not in ('u:', 'p:', 's:') for u, c in mapping.items()):
        raise ValueError('Expected role-specific single-letter key')
    started = time.monotonic()
    lookup = {c: i for i, c in enumerate(prior.alphabet)}
    costs = choice_costs(mapping)
    # State=(last order-1 letters, total letters); value=(bits, linked parse path).
    states = {('', 0): (0., None)}
    for token in tokens:
        readings = options(token, mapping)
        candidates = {}
        for (context, length), (bits, path) in states.items():
            for parts, units in readings:
                history, extra = context, 0.
                for unit in units:
                    letter = mapping[unit]
                    index = 0
                    for c in history + letter:
                        index = index * prior.base + lookup[c]
                    extra += float(prior.logs[len(history)][index]) + costs[unit]
                    history = (history + letter)[-(prior.order - 1):] if prior.order > 1 else ''
                state = (history, length + len(units))
                value = bits + extra
                if state not in candidates or value < candidates[state][0]:
                    candidates[state] = (value, (path, parts))
        states = dict(sorted(candidates.items(), key=lambda kv: (kv[1][0] + integer_bits(kv[0][1]), kv[0]))[:width])
    state, (bits, path) = min(states.items(), key=lambda kv: (kv[1][0] + integer_bits(kv[0][1]), kv[0]))
    segmentation = []
    while path is not None:
        path, parts = path
        segmentation.append(parts)
    segmentation.reverse()
    units = [r + ':' + p for parts in segmentation for r, p in zip(('u',) if len(parts) == 1 else ('p', 's'), parts)]
    return dict(segmentation=segmentation, recovered=''.join(mapping[u] for u in units),
                bits=bits + integer_bits(state[1]), width=width, seconds=time.monotonic() - started)
