"""Lexicon repair with a self-inclusive concatenation test.

Round four's test (`voynich/lexicon_repair.py`) compares a whole piece's token count with the
count its halves would produce as a prefix+suffix bigram. The bigram count is taken from split
parses only. So a spurious concatenation that the EM already parses as one whole piece removes its
own occurrences from the bigram it is tested against: its expectation shrinks, its ratio grows, and
it survives. Under the concatenation hypothesis those whole-parsed occurrences are that bigram, so
here they are added back, to the bigram count and to the half-piece unit counts. Everything else is
unchanged.
"""
from collections import Counter
import time

from voynich.joint_segments_v2 import joint_em
from voynich.lexicon_repair import _units, _key, unparsed_complements


def concatenation_ratios(tokens, pieces, segmentation, recovered):
    units = _units(segmentation); key = _key(units, recovered)
    unit_count = Counter(units); role_letter = Counter((u[0], key[u]) for u in units)
    bigrams = Counter(); whole = Counter(); pos = 0
    for parts in segmentation:
        if len(parts) == 2:
            bigrams[recovered[pos:pos + 2]] += 1
        else:
            whole[parts[0]] += 1
        pos += len(parts)
    token_count = Counter(tokens); ratios = {}
    for t in pieces:
        if t not in token_count:
            continue
        n = whole[t]; expected = 0.
        for i in range(1, len(t)):
            a, b = t[:i], t[i:]
            if a in pieces and b in pieces and f'p:{a}' in key and f's:{b}' in key:
                x, z = key[f'p:{a}'], key[f's:{b}']
                expected = max(expected, (bigrams[x + z] + n) * (unit_count[f'p:{a}'] + n) / (role_letter[('p', x)] + n)
                               * (unit_count[f's:{b}'] + n) / (role_letter[('s', z)] + n))
        if expected > 0:
            ratios[t] = token_count[t] / expected
    return ratios


def repair(tokens, prior, result, theta=5., complement_minimum=2, passes=1, usage_floor=1., **em):
    """Round four's repair loop with the self-inclusive ratio."""
    started = time.monotonic()
    pieces = set(result['candidate_pieces']); current = result; record = []
    for p in range(passes):
        pruned = set() if p == 0 else {q for q in pieces if current['piece_usage'].get(q, 0.) < usage_floor}
        pieces -= pruned
        ratios = concatenation_ratios(tokens, pieces, current['segmentation'], current['recovered'])
        dropped = {t for t, r in ratios.items() if r < theta}
        pieces -= dropped
        admitted = unparsed_complements(tokens, pieces, complement_minimum)
        pieces |= admitted
        current = joint_em(tokens, prior, pieces=pieces, **em)
        record.append(dict(usage_pruned=len(pruned), concatenations_dropped=len(dropped), complements_admitted=len(admitted),
                           pieces=len(pieces), log_likelihood=current['log_likelihood'], cap_hit=current['cap_hit']))
    current.update(repair=dict(theta=theta, complement_minimum=complement_minimum, passes=passes, usage_floor=usage_floor, record=record,
                               self_inclusive=True, seconds=time.monotonic() - started))
    return current
