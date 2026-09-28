"""Propose rare half pieces by the same leave-one-out context test as whole-token admission.

A token whose true split is (a, b) with b missing from the key is read as one unknown whole, or split
at the wrong point. For each occurrence and split point where exactly one half is a known key unit in
its role, the other half is a candidate new unit: the reading becomes the known letter plus a new
letter. As in `voynich/whole_admission.py`, the saving over the current reading is summed over every
occurrence of the candidate, the new letter is chosen from the other occurrences, and a candidate is
proposed when that leave-one-out saving exceeds the key-entry bits plus log2 of the number tested.

Proposals are lexicon strings, not key entries: joint EM decides whether and how they are used.
Reads only ciphertext and the decoder's own output.
"""
import math

import numpy as np

from voynich.piece_admission import _cond_bits
from voynich.whole_admission import _letters, leave_one_out


def half_savings(tokens, segmentation, mapping, prior, context=4):
    """Per candidate unit ('p:' or 's:' + piece): array (occurrences, letters) of bit savings."""
    lookup = {c: i for i, c in enumerate(prior.alphabet)}
    letters = [_letters(parts, mapping) for parts in segmentation]
    rows = {}
    for i, token in enumerate(tokens):
        before = ''.join(letters[max(0, i - context):i])[-context:]
        after = ''.join(letters[i + 1:i + 1 + context])[:context]
        current = None
        for j in range(1, len(token)):
            prefix, suffix = 'p:' + token[:j], 's:' + token[j:]
            if (prefix in mapping) == (suffix in mapping):
                continue
            if current is None:
                current = _cond_bits(prior, lookup, before, letters[i] + after)
            if prefix in mapping:
                unit, reading = suffix, lambda y: mapping[prefix] + y
            else:
                unit, reading = prefix, lambda y: y + mapping[suffix]
            rows.setdefault(unit, []).append(
                [current - _cond_bits(prior, lookup, before, reading(y) + after) for y in prior.alphabet])
    return {unit: np.array(r) for unit, r in rows.items()}


def admit_halves(tokens, segmentation, mapping, prior, context=4):
    """Return ({unit: letter}, record) for proposed half units seen at least twice."""
    key_bits = 1. + math.ceil(math.log2(prior.base))
    tested = {u: s for u, s in half_savings(tokens, segmentation, mapping, prior, context).items() if len(s) >= 2}
    threshold = key_bits + math.log2(max(2, len(tested)))
    admitted = {}
    for unit, saving in tested.items():
        if leave_one_out(saving) > threshold:
            admitted[unit] = prior.alphabet[int(np.argmax(saving.sum(axis=0)))]
    return admitted, dict(tested=len(tested), threshold=threshold, admitted=len(admitted))
