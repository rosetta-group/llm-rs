"""Admit rare cipher pieces by context, not frequency.

After a reparse, some true pieces are still missing from the lexicon: they occur a few times, never
pass a count threshold, and their tokens get read with the wrong pieces. For each rare token, this
module considers readings that use exactly one new piece:
  - the whole token as a new one-letter piece,
  - a prefix+suffix split where one half is a known piece in its role and the other half is new.
For every (role, piece) candidate and every letter it could stand for, it sums the change in
character bits (order-n prior, local context of the neighbouring letters) plus homophone-choice bits
over the candidate's occurrences. Only occurrences that improve count. A candidate is admitted with
its best letter when the total saving reaches `threshold` bits. Reads only ciphertext and the
decoder's own output.
"""
from collections import Counter, defaultdict
import math


def _cond_bits(prior, lookup, before, text):
    """Bits of `text` given the letters `before` (at most order-1 used)."""
    history = before[-(prior.order - 1):] if prior.order > 1 else ''
    bits = 0.
    for c in text:
        index = 0
        for x in history + c:
            index = index * prior.base + lookup[x]
        bits += float(prior.logs[len(history)][index])
        history = (history + c)[-(prior.order - 1):] if prior.order > 1 else ''
    return bits


def _units(parts):
    return ['u:' + parts[0]] if len(parts) == 1 else ['p:' + parts[0], 's:' + parts[1]]


def admit(tokens, segmentation, mapping, prior, max_count=5, threshold=10., context=4):
    """Return (new_mapping, record). `segmentation` and `mapping` are the current parse and role key."""
    lookup = {c: i for i, c in enumerate(prior.alphabet)}
    letters = [''.join(mapping[u] for u in _units(parts)) for parts in segmentation]
    starts, pos = [], 0
    for l in letters:
        starts.append(pos); pos += len(l)
    text = ''.join(letters)
    per_role = Counter((u[0], c) for u, c in mapping.items())
    choice = lambda u, c: math.log2(per_role[(u[0], c)]) if u in mapping else math.log2(per_role[(u[0], c)] + 1)
    counts = Counter(tokens)
    gains = defaultdict(lambda: defaultdict(float))  # (role:piece) -> letter -> summed saving
    occurrences = Counter()
    for j, (token, parts) in enumerate(zip(tokens, segmentation)):
        if counts[token] > max_count:
            continue
        before = text[max(0, starts[j] - context):starts[j]]
        after = text[starts[j] + len(letters[j]):starts[j] + len(letters[j]) + context]
        old_units = _units(parts)
        old_bits = _cond_bits(prior, lookup, before, letters[j] + after) + sum(choice(u, mapping[u]) for u in old_units)
        readings = []
        if 'u:' + token not in mapping:
            readings.append(('u:' + token, lambda L: [('u:' + token, L)]))
        for i in range(1, len(token)):
            a, b = 'p:' + token[:i], 's:' + token[i:]
            if a in mapping and b not in mapping:
                readings.append((b, lambda L, a=a, b=b: [(a, mapping[a]), (b, L)]))
            if b in mapping and a not in mapping:
                readings.append((a, lambda L, a=a, b=b: [(a, L), (b, mapping[b])]))
        for key, build in readings:
            occurrences[key] += 1
            for L in prior.alphabet:
                units = build(L)
                new_bits = _cond_bits(prior, lookup, before, ''.join(c for _, c in units) + after) + sum(choice(u, c) for u, c in units)
                if new_bits < old_bits:
                    gains[key][L] += old_bits - new_bits
    admitted = {}
    for key, by_letter in gains.items():
        L, saving = max(by_letter.items(), key=lambda kv: (kv[1], kv[0]))
        if saving >= threshold:
            admitted[key] = (L, saving)
    new_mapping = dict(mapping)
    for key, (L, _) in admitted.items():
        new_mapping[key] = L
    record = dict(candidates=len(gains), admitted=len(admitted), threshold=threshold, max_count=max_count,
                  admitted_units=sorted(admitted), savings={k: round(v[1], 2) for k, v in sorted(admitted.items())},
                  occurrences={k: occurrences[k] for k in sorted(admitted)})
    return new_mapping, record
