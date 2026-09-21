"""Round-two joint EM: candidate-lexicon override, expected piece usage, and usage pruning.

Keeps voynich/joint_segments.py byte-identical to the round-one freeze. The forward-backward,
candidate rule and parse builder are imported unchanged; only the EM driver is extended.
"""
from collections import Counter
import time

import numpy as np

from voynich.joint_segments import forward_backward, candidate_pieces, build_parses


def joint_em(tokens, prior, minimum=3, restarts=4, iterations=60, smoothing=.1, seed=0, cap=1800, warm_start=None, warm_mass=.9,
             pieces=None):
    """Run EM over latent parses and emissions; return the best restart's decode.

    `warm_start` is a role-unit key ('u:piece' -> letter); when given, the first restart
    initializes each known unit's emission with `warm_mass` on that letter instead of a
    random table, so EM re-estimates parses around an already refined key. `pieces`
    overrides the frequency-based candidate lexicon.
    """
    started = time.monotonic()
    base = prior.base
    pieces = candidate_pieces(tokens, minimum) if pieces is None else set(pieces)
    inventory, (starts, lengths, first, second) = build_parses(tokens, pieces)
    index = {p: i for i, p in enumerate(inventory)}
    unigram = prior.probabilities[0]
    bigram = prior.probabilities[1].reshape(base, base)
    tri = prior.probabilities[2].reshape(base, base, base)
    tri = (tri + bigram[None, :, :] + unigram[None, None, :]) / 3.
    init = np.outer(unigram, unigram)
    rng = np.random.default_rng(seed)
    best = None; history = []
    for r in range(restarts):
        if r and time.monotonic() - started > cap:
            break
        e_u = rng.uniform(size=(base, len(inventory))); e_u /= e_u.sum(axis=1, keepdims=True)
        e_p = rng.uniform(size=(base, len(inventory))); e_p /= e_p.sum(axis=1, keepdims=True)
        e_s = rng.uniform(size=(base, len(inventory))); e_s /= e_s.sum(axis=1, keepdims=True)
        if warm_start is not None and r == 0:
            tables = {'u': e_u, 'p': e_p, 's': e_s}
            for unit, letter in warm_start.items():
                role, piece = unit.split(':', 1)
                if piece in index:
                    column = tables[role][:, index[piece]]
                    column *= (1. - warm_mass) / column.sum()
                    column[prior.alphabet.index(letter)] += warm_mass
            for table in tables.values():
                table /= table.sum(axis=1, keepdims=True)
        loglik = -np.inf
        for it in range(iterations):
            cu, cp, cs, parse_post, lp1, lp2, loglik = forward_backward(starts, lengths, first, second, base, tri, e_u, e_p, e_s, init)
            e_u = cu + smoothing; e_u /= e_u.sum(axis=1, keepdims=True)
            e_p = cp + smoothing; e_p /= e_p.sum(axis=1, keepdims=True)
            e_s = cs + smoothing; e_s /= e_s.sum(axis=1, keepdims=True)
            if time.monotonic() - started > cap:
                break
        cu, cp, cs, parse_post, lp1, lp2, loglik = forward_backward(starts, lengths, first, second, base, tri, e_u, e_p, e_s, init)
        history.append(float(loglik))
        if best is None or loglik > best['log_likelihood']:
            # decode: best parse per token, then argmax letters within it
            recovered = []; chosen = []
            for t in range(len(tokens)):
                span = range(starts[t], starts[t + 1])
                pi = max(span, key=lambda i: parse_post[i])
                chosen.append((inventory[first[pi]],) if lengths[pi] == 1 else (inventory[first[pi]], inventory[second[pi]]))
                recovered.append(prior.alphabet[int(np.argmax(lp1[pi]))])
                if lengths[pi] == 2:
                    recovered.append(prior.alphabet[int(np.argmax(lp2[pi]))])
            usage = Counter()
            for t in range(len(tokens)):
                for pi in range(starts[t], starts[t + 1]):
                    usage[inventory[first[pi]]] += parse_post[pi]
                    if lengths[pi] == 2:
                        usage[inventory[second[pi]]] += parse_post[pi]
            best = dict(recovered=''.join(recovered), segmentation=chosen, log_likelihood=float(loglik), best_restart=r,
                        pieces=len(inventory), parses=int(len(lengths)), candidate_pieces=pieces,
                        piece_usage={p: float(u) for p, u in usage.items()})
    best.update(restarts=len(history), restart_scores=history, iterations=iterations, minimum=minimum,
                seconds=time.monotonic() - started, cap_hit=time.monotonic() - started > cap)
    return best


def prune_and_rerun(tokens, prior, first_pass, minimum_usage=3., **em):
    """Drop candidate pieces whose expected usage in `first_pass` is below `minimum_usage`,
    rebuild the parses, and run EM again on the pruned lexicon. Tokens that lose every parse
    keep their whole-token fallback. Returns the second pass result with the pruning record."""
    kept = {p for p in first_pass['candidate_pieces'] if first_pass['piece_usage'].get(p, 0.) >= minimum_usage}
    second = joint_em(tokens, prior, pieces=kept, **em)
    second.update(pruned_from=len(first_pass['candidate_pieces']), pruned_to=len(kept), minimum_usage=minimum_usage)
    return second
