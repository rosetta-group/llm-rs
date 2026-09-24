"""Acceptance decisions and transfer using an immutable, role-specific cipher key."""
import math

from voynich.context_reparse import options, reparse

EXCESS_CEILING = .50
MIN_MARGIN = .25
MIN_COVERAGE = .95


def transfer(tokens, mapping, prior, width=128):
    """Decode covered runs only. Unknown tokens are explicit gaps, never new key entries."""
    before = dict(mapping)
    runs, pending, covered, glyphs, recovered = [], [], 0, 0, []

    def flush():
        if pending:
            result = reparse(pending, mapping, prior, width=width)
            runs.append(result['recovered'])
            recovered.append(result['recovered'])
            pending.clear()

    for token in tokens:
        try:
            options(token, mapping)
        except ValueError:
            flush()
            recovered.append('?')
        else:
            pending.append(token)
            covered += 1
            glyphs += len(token)
    flush()
    assert mapping == before, 'Transfer mutated the key'
    letters = sum(map(len, runs))
    return dict(recovered=''.join(recovered), covered_tokens=covered, tokens=len(tokens),
                token_coverage=covered / len(tokens) if tokens else 0.,
                glyph_coverage=glyphs / sum(map(len, tokens)) if tokens else 0.,
                recovered_letters=letters, covered_runs=len(runs),
                bits_per_letter=sum(prior.bits(r) for r in runs) / letters if letters else None)


def rank(rows, field):
    ordered = sorted((r[field], language) for language, r in rows.items()
                     if r[field] is not None and math.isfinite(r[field]))
    if len(ordered) < 2:
        return dict(winner=None, excess=None, margin=None)
    return dict(winner=ordered[0][1], excess=ordered[0][0], margin=ordered[1][0] - ordered[0][0])


def decide(rows, candidates=None):
    """Rank first; never select a different winner merely because it clears a gate."""
    rows = {l: r for l, r in rows.items() if candidates is None or l in candidates}
    fit, held = rank(rows, 'fit_excess'), rank(rows, 'transfer_excess')
    reasons = []
    cap = any(r['cap_hit'] for r in rows.values())
    if cap:
        reasons.append('compute_cap')
    for name, ranking in [('fit', fit), ('transfer', held)]:
        if ranking['winner'] is None:
            reasons.append(name + '_unscorable')
        else:
            if ranking['excess'] > EXCESS_CEILING:
                reasons.append(name + '_excess')
            if ranking['margin'] < MIN_MARGIN:
                reasons.append(name + '_margin')
    if fit['winner'] != held['winner']:
        reasons.append('winner_changed')
    if held['winner'] is not None and rows[held['winner']]['coverage'] < MIN_COVERAGE:
        reasons.append('coverage')
    accepted = fit['winner'] if not reasons else None
    in_sample = fit['winner'] if not cap and fit['winner'] is not None and \
        fit['excess'] <= EXCESS_CEILING and fit['margin'] >= MIN_MARGIN else None
    return dict(accepted=accepted, in_sample_accepted=in_sample, fit=fit, transfer=held,
                reasons=reasons, inconclusive=cap)


def stop_reason(outcomes):
    if any(r['decision']['inconclusive'] for r in outcomes):
        return 'inconclusive_compute_cap'
    if any(r['kind'] == 'positive' and r['decision']['accepted'] not in (None, r['language']) for r in outcomes):
        return 'wrong_language_accepted'
    if any(r['kind'] != 'positive' and r['decision']['accepted'] is not None for r in outcomes):
        return 'negative_accepted'
    if sum(r['kind'] == 'positive' and r['decision']['accepted'] is None for r in outcomes) >= 2:
        return 'two_positives_rejected'
    return None
