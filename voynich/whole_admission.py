"""Admit rare whole-token cipher pieces by a leave-one-out context test.

After the frozen fit, most missing true pieces are one-letter whole tokens that occur fewer
times than the candidate minimum. Each such token also splits into two known pieces, so it is
read as two letters. Earlier context admission (`voynich/piece_admission.py`) summed only the
occurrences that improved and chose the best of all letters, which found savings by chance.

This test considers one reading per token type: the whole token as a new one-letter piece. For
each occurrence it computes the character-bit saving of each letter over the current split, in
the local context of the decoded neighbours. The letter for an occurrence is chosen from the
other occurrences only, and every occurrence counts, improving or not. A type is admitted when
that leave-one-out saving exceeds the key-entry bits plus log2 of the number of types tested.
Types seen once cannot be held out and are never admitted. Reads only ciphertext and the
decoder's own output.
"""
import math

import numpy as np

from voynich.piece_admission import _cond_bits


def _letters(parts, mapping):
    return ''.join(mapping[u] for u in (['u:' + parts[0]] if len(parts) == 1 else ['p:' + parts[0], 's:' + parts[1]]))


def whole_savings(tokens, segmentation, mapping, prior, context=4):
    """Per split token type without a whole key entry: array (occurrences, letters) of bit savings."""
    lookup = {c: i for i, c in enumerate(prior.alphabet)}
    letters = [_letters(parts, mapping) for parts in segmentation]
    positions = {}
    for i, (token, parts) in enumerate(zip(tokens, segmentation)):
        if len(parts) == 2 and 'u:' + token not in mapping:
            positions.setdefault(token, []).append(i)
    savings = {}
    for token, where in positions.items():
        rows = []
        for i in where:
            before = ''.join(letters[max(0, i - context):i])[-context:]
            after = ''.join(letters[i + 1:i + 1 + context])[:context]
            split = _cond_bits(prior, lookup, before, letters[i] + after)
            rows.append([split - _cond_bits(prior, lookup, before, y + after) for y in prior.alphabet])
        savings[token] = np.array(rows)
    return savings


def leave_one_out(saving):
    total = saving.sum(axis=0)
    return float(sum(row[np.argmax(total - row)] for row in saving))


def admit_wholes(tokens, segmentation, mapping, prior, context=4):
    """Return ({token: letter}, record). Letters are the argmax over all occurrences."""
    key_bits = 1. + math.ceil(math.log2(prior.base))
    tested = {t: s for t, s in whole_savings(tokens, segmentation, mapping, prior, context).items() if len(s) >= 2}
    threshold = key_bits + math.log2(max(2, len(tested)))
    admitted, scores = {}, {}
    for token, saving in tested.items():
        scores[token] = leave_one_out(saving)
        if scores[token] > threshold:
            admitted[token] = prior.alphabet[int(np.argmax(saving.sum(axis=0)))]
    return admitted, dict(tested=len(tested), threshold=threshold, admitted=len(admitted))


def apply_wholes(tokens, segmentation, admitted):
    return [(t,) if t in admitted else tuple(parts) for t, parts in zip(tokens, segmentation)]
