"""Move token split points when the whole lexicon becomes cheaper to describe.

Development diagnostics found the decoder's main remaining error is a systematic split point: a
glyph run that begins the true second piece is attached to the first (`l`+`chdy` read as
`lch`+`dy`). The key compensates on the fitted passage, so letter-level tests see little gain, but
the extra first pieces (`lch`, `qoch`, `tch`, ...) and the overloaded second piece (`dy`) cost key
entries and letters, and they fail on new text.

A move introduces one piece in one role by shifting the split of every token that would use it,
provided the other half is already a key unit in its role:

  new second piece x+b:  (a+x, b) -> (a, x+b)   when p:a is known
  new first piece  a+x:  (a, x+b) -> (a+x, b)   when s:b is known

The new unit takes the letter the decoder already reads at that position (majority), the known half
keeps its letter, and the move is scored by the key-search description length (`objective`: letter
bits, key-entry bits and homophone-choice bits). The cheapest move is applied while any move lowers
the total. Reads only ciphertext and the decoder's own output.
"""
from collections import Counter, defaultdict

from voynich.variable_units import objective


def _units(parts):
    return ['u:' + parts[0]] if len(parts) == 1 else ['p:' + parts[0], 's:' + parts[1]]


def candidates(segmentation, mapping, minimum=2):
    """{move: [(token index, new parts, new unit, position of its letter)]} for moves used at least `minimum` times.

    A move is either one new unit ('unit', 's:chdy'), or one glyph run shifted across every token that
    allows it ('run', 'to_second', 'ch'), which introduces several new units at once and can retire
    first pieces such as `lch` and `qoch` together.
    """
    moves = defaultdict(list)
    for i, parts in enumerate(segmentation):
        if len(parts) != 2:
            continue
        a, b = parts
        for k in range(1, len(a)):
            head, x = a[:-k], a[-k:]
            unit = 's:' + x + b
            if 'p:' + head in mapping and unit not in mapping:
                moves[('unit', unit)].append((i, (head, x + b), unit, 1))
            if 'p:' + head in mapping:
                moves[('run', 'to_second', x)].append((i, (head, x + b), unit, 1))
        for k in range(1, len(b)):
            x, tail = b[:k], b[k:]
            unit = 'p:' + a + x
            if 's:' + tail in mapping and unit not in mapping:
                moves[('unit', unit)].append((i, (a + x, tail), unit, 0))
            if 's:' + tail in mapping:
                moves[('run', 'to_first', x)].append((i, (a + x, tail), unit, 0))
    out = {}
    for move, items in moves.items():
        seen = {}
        for item in items:
            seen.setdefault(item[0], item)   # one reading per token: the first (shortest shift) found
        if len(seen) >= minimum:
            out[move] = list(seen.values())
    return out


def apply(segmentation, mapping, moves):
    letters = [''.join(mapping[u] for u in _units(parts)) for parts in segmentation]
    votes = defaultdict(Counter)
    for i, _, unit, position in moves:
        votes[unit][letters[i][position]] += 1
    new_segmentation = list(segmentation)
    for i, parts, _, _ in moves:
        new_segmentation[i] = parts
    key = dict(mapping)
    for unit, count in votes.items():
        if unit not in key:
            key[unit] = count.most_common(1)[0][0]
    used = {u for parts in new_segmentation for u in _units(parts)}
    return new_segmentation, {u: c for u, c in key.items() if u in used}


def shift(segmentation, mapping, prior, max_moves=200, minimum=2):
    """Greedy description-length descent over split-point moves. Returns (segmentation, key, record)."""
    segmentation = [tuple(p) for p in segmentation]
    key = {u: c for u, c in mapping.items() if u in {x for p in segmentation for x in _units(p)}}
    current = objective([u for p in segmentation for u in _units(p)], prior, key)
    record = []
    for _ in range(max_moves):
        best = None
        for move, items in candidates(segmentation, key, minimum).items():
            seg, k = apply(segmentation, key, items)
            score = objective([u for p in seg for u in _units(p)], prior, k)
            if score < current and (best is None or score < best[0]):
                best = (score, move, seg, k, len(items))
        if best is None:
            break
        record.append(dict(move=list(best[1]), tokens=best[4], saving=current - best[0]))
        current, segmentation, key = best[0], best[2], best[3]
    return segmentation, key, dict(moves=record, bits=current)
