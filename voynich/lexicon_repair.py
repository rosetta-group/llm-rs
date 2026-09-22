"""Lexicon repair for the joint segmentation EM (round four).

The joint EM of `voynich/joint_segments.py` parses tokens with a candidate lexicon built from
raw string frequency. Development analysis with the true lexicon as an oracle found the same
EM reaches 97% parse agreement and 0.5% character error when given the right pieces, so the
lexicon, not the model, carries the remaining error. It has two defects:

* Concatenations admitted as whole pieces. A frequent plaintext bigram produces the same
  prefix+suffix string again and again; once that string passes the frequency threshold it
  becomes a whole-piece candidate, and the one-letter derivation then beats the two-letter one.
  The test here compares a whole piece's token count with the count its two halves would
  produce as a bigram under the decoder's own key and text; a true one-letter piece is far more
  frequent than that expectation, a concatenation is not.
* Rare pieces never admitted. Pieces of rare letters occur one to four times and never reach
  the threshold; their tokens fall back to a whole parse of an unknown string. For a token that
  has no parse at all, the complement of a known half is admitted as a candidate.

Both repairs read only the ciphertext and the decoder's previous output.
"""
from collections import Counter, defaultdict
import time

from voynich.joint_segments_v2 import joint_em


def _units(segmentation):
    return [f'{role}:{piece}' for parts in segmentation for piece, role in zip(parts, ('u',) if len(parts) == 1 else ('p', 's'))]


def _key(units, letters):
    votes = defaultdict(Counter)
    for u, c in zip(units, letters):
        votes[u][c] += 1
    return {u: votes[u].most_common(1)[0][0] for u in votes}


def concatenation_ratios(tokens, pieces, segmentation, recovered):
    """For each piece that occurs as a whole token and splits into two pieces, the ratio of its
    token count to the count expected if it were a prefix+suffix bigram under the decoded key.

    Expected count = bigram count in the recovered letters × share of the first letter's prefix
    occurrences carried by the first half × share of the second letter's suffix occurrences carried
    by the second half. The largest expectation over split points is used. Pieces whose halves have
    no decoded prefix or suffix letter are not scored.
    """
    units = _units(segmentation); key = _key(units, recovered)
    unit_count = Counter(units); role_letter = Counter()
    for u in units:
        role_letter[(u[0], key[u])] += 1
    bigrams = Counter(); pos = 0
    for parts in segmentation:
        if len(parts) == 2:
            bigrams[recovered[pos:pos + 2]] += 1
        pos += len(parts)
    token_count = Counter(tokens); ratios = {}
    for t in pieces:
        if t not in token_count:
            continue
        expected = 0.
        for i in range(1, len(t)):
            a, b = t[:i], t[i:]
            if a in pieces and b in pieces and f'p:{a}' in key and f's:{b}' in key:
                x, z = key[f'p:{a}'], key[f's:{b}']
                expected = max(expected, bigrams[x + z] * unit_count[f'p:{a}'] / max(1, role_letter[('p', x)])
                               * unit_count[f's:{b}'] / max(1, role_letter[('s', z)]))
        if expected > 0:
            ratios[t] = token_count[t] / expected
    return ratios


def unparsed_complements(tokens, pieces, minimum=2):
    """Strings that complete a known half of a token that has no whole piece and no split,
    counted over such tokens; those reaching `minimum` occurrences are returned."""
    token_count = Counter(tokens); complements = Counter()
    for t, n in token_count.items():
        if t in pieces or any(t[:i] in pieces and t[i:] in pieces for i in range(1, len(t))):
            continue
        for i in range(1, len(t)):
            if t[:i] in pieces and t[i:] not in pieces:
                complements[t[i:]] += n
            if t[i:] in pieces and t[:i] not in pieces:
                complements[t[:i]] += n
    return {p for p, n in complements.items() if n >= minimum}


def repair(tokens, prior, result, theta=5., complement_minimum=2, passes=1, usage_floor=1., **em):
    """Repair the candidate lexicon of a finished EM `result` and run EM again, `passes` times.

    Each pass drops whole pieces whose concatenation ratio is below `theta`, admits complements
    for unparsed tokens (`complement_minimum` occurrences), and reruns EM on the new lexicon. From the second pass on, pieces whose
    expected usage in the previous pass fell below `usage_floor` are dropped first. Returns the
    last EM result with a `repair` record of every pass.
    """
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
                               seconds=time.monotonic() - started))
    return current
