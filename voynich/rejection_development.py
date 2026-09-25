"""Development-only rejection rule and frequency-preserving copying control."""
from collections import Counter
import math
import random

from voynich.rejection import decide


def decide_transfer(rows, candidates=None, transfer_ceiling=.5):
    if not math.isfinite(transfer_ceiling) or transfer_ceiling < 0:
        raise ValueError('A finite nonnegative transfer ceiling is required')
    original = decide(rows, candidates)
    reasons = [r for r in original['reasons'] if r not in ('fit_excess', 'transfer_excess')]
    ranking = original['transfer']
    if ranking['winner'] is not None and ranking['excess'] > transfer_ceiling:
        reasons.append('transfer_excess')
    return dict(accepted=original['fit']['winner'] if not reasons else None,
                fit=original['fit'], transfer=ranking, reasons=reasons,
                inconclusive=original['inconclusive'], variant='transfer_centered',
                transfer_ceiling=transfer_ceiling)


def frequency_copy(tokens, seed, copy_probability=.8, window=50):
    """Consume the exact input multiset while preferentially copying recent positions."""
    if not 0 <= copy_probability <= 1 or window < 1:
        raise ValueError('Valid copy probability and positive window required')
    remaining = Counter(tokens)
    inventory = sorted(remaining)
    rng = random.Random(seed)
    output = []
    for left in range(len(tokens), 0, -1):
        recent = [token for token in output[-window:] if remaining[token]]
        if rng.random() < copy_probability and recent:
            token = rng.choice(recent)
        else:
            draw = rng.randrange(left)
            for token in inventory:
                draw -= remaining[token]
                if draw < 0:
                    break
        output.append(token)
        remaining[token] -= 1
    assert not any(remaining.values())
    return output


def copying_diagnostics(tokens, window=50):
    return dict(tokens=len(tokens), types=len(set(tokens)),
                adjacent_repeat_rate=sum(a == b for a, b in zip(tokens, tokens[1:])) / max(1, len(tokens) - 1),
                recent_repeat_rate=sum(t in tokens[max(0, i-window):i] for i, t in enumerate(tokens)) / max(1, len(tokens)))
