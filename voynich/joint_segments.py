"""Joint segmentation and decipherment of tokens that hide one or two letters.

A trigram hidden Markov model over plaintext letters, where each observed
token is emitted by one of several latent parses: the whole token as one
piece standing for one letter, or a split into two pieces standing for two
letters. Emission tables are separate for the three piece roles (whole,
first, second). EM learns the emissions and the parse posteriors together,
so segmentation errors are not frozen before decipherment.

This generalizes the fixed-segmentation trigram HMM of Berg-Kirkpatrick and
Klein (2013) to latent token parses. Language data, candidate-piece rules,
iteration counts and CPU caps belong to the experiment protocol.
"""
from collections import Counter
import time

import numpy as np
from numba import njit


@njit(cache=True)
def forward_backward(parse_start, parse_len, parse_piece1, parse_piece2, base, tri, e_u, e_p, e_s, init):
    """Scaled forward/backward over tokens with latent parses.

    parse_start[t]..parse_start[t+1] index the parses of token t. A parse with
    parse_len 1 has piece1 (role whole); with parse_len 2 pieces (first, second).
    tri[x1, x2, y] is p(y | x1, x2); init[x1, x2] the distribution of the first
    two letters. Returns expected emission counts, per-parse posteriors and the
    log likelihood.
    """
    tokens = len(parse_start) - 1
    alpha = np.zeros((tokens + 1, base, base))
    scale = np.zeros(tokens + 1)
    # State before any token: a virtual pair of boundary letters is avoided by
    # starting from the prior over the first two letters once two letters exist.
    # We handle the start by a uniform state and let init act on the first emitted letters.
    alpha[0] = 1. / (base * base)
    scale[0] = 1.
    beta_tmp = np.zeros((base, base))
    for t in range(tokens):
        out = np.zeros((base, base))
        for pi in range(parse_start[t], parse_start[t + 1]):
            if parse_len[pi] == 1:
                q = parse_piece1[pi]
                for x2 in range(base):
                    for y in range(base):
                        s = 0.
                        for x1 in range(base):
                            s += alpha[t, x1, x2] * tri[x1, x2, y]
                        out[x2, y] += s * e_u[y, q]
            else:
                q1 = parse_piece1[pi]; q2 = parse_piece2[pi]
                for x2 in range(base):
                    for a in range(base):
                        s = 0.
                        for x1 in range(base):
                            s += alpha[t, x1, x2] * tri[x1, x2, a]
                        beta_tmp[x2, a] = s * e_p[a, q1]
                for a in range(base):
                    for b in range(base):
                        s = 0.
                        for x2 in range(base):
                            s += beta_tmp[x2, a] * tri[x2, a, b]
                        out[a, b] += s * e_s[b, q2]
        z = out.sum()
        scale[t + 1] = z
        alpha[t + 1] = out / z
    loglik = np.log(scale[1:]).sum()
    # backward
    beta = np.ones((base, base))
    counts_u = np.zeros_like(e_u); counts_p = np.zeros_like(e_p); counts_s = np.zeros_like(e_s)
    parse_post = np.zeros(len(parse_len))
    letter_post1 = np.zeros((len(parse_len), base))
    letter_post2 = np.zeros((len(parse_len), base))
    for t in range(tokens - 1, -1, -1):
        new_beta = np.zeros((base, base))
        for pi in range(parse_start[t], parse_start[t + 1]):
            if parse_len[pi] == 1:
                q = parse_piece1[pi]
                mass = 0.
                for x1 in range(base):
                    for x2 in range(base):
                        acc = 0.
                        for y in range(base):
                            w = tri[x1, x2, y] * e_u[y, q] * beta[x2, y]
                            acc += w
                            g = alpha[t, x1, x2] * w
                            letter_post1[pi, y] += g
                            mass += g
                        new_beta[x1, x2] += acc
                parse_post[pi] = mass / scale[t + 1]
            else:
                q1 = parse_piece1[pi]; q2 = parse_piece2[pi]
                # gamma over (x2, a): sum_x1 alpha * tri(a) * e_p
                for x2 in range(base):
                    for a in range(base):
                        s = 0.
                        for x1 in range(base):
                            s += alpha[t, x1, x2] * tri[x1, x2, a]
                        beta_tmp[x2, a] = s * e_p[a, q1]
                mass = 0.
                # forward part for posteriors
                for x2 in range(base):
                    for a in range(base):
                        acc = 0.
                        for b in range(base):
                            w = tri[x2, a, b] * e_s[b, q2] * beta[a, b]
                            acc += w
                            g = beta_tmp[x2, a] * w
                            letter_post1[pi, a] += g
                            letter_post2[pi, b] += g
                            mass += g
                        # backward contribution to previous states through x1 is handled below
                        beta_tmp[x2, a] = acc  # reuse as partial backward over (x2,a): sum_b tri*e_s*beta
                for x1 in range(base):
                    for x2 in range(base):
                        acc = 0.
                        for a in range(base):
                            acc += tri[x1, x2, a] * e_p[a, q1] * beta_tmp[x2, a]
                        new_beta[x1, x2] += acc
                parse_post[pi] = mass / scale[t + 1]
        beta = new_beta / scale[t + 1]
    # normalize letter posteriors per parse and accumulate emission counts
    for pi in range(len(parse_len)):
        total = letter_post1[pi].sum()
        if total > 0:
            letter_post1[pi] /= total
            letter_post1[pi] *= parse_post[pi]
        if parse_len[pi] == 1:
            for y in range(base):
                counts_u[y, parse_piece1[pi]] += letter_post1[pi, y]
        else:
            total2 = letter_post2[pi].sum()
            if total2 > 0:
                letter_post2[pi] /= total2
                letter_post2[pi] *= parse_post[pi]
            for y in range(base):
                counts_p[y, parse_piece1[pi]] += letter_post1[pi, y]
                counts_s[y, parse_piece2[pi]] += letter_post2[pi, y]
    return counts_u, counts_p, counts_s, parse_post, letter_post1, letter_post2, loglik


def candidate_pieces(tokens, minimum=3):
    """Strings that occur at least `minimum` times as a whole token, a proper prefix or a proper suffix."""
    counts = Counter(tokens)
    seen = Counter()
    for t, n in counts.items():
        seen[t] += n
        for i in range(1, len(t)):
            seen[t[:i]] += n
            seen[t[i:]] += n
    return {p for p, n in seen.items() if n >= minimum}


def build_parses(tokens, pieces):
    """Splits need both parts in `pieces`; a whole parse needs the token itself in
    `pieces`, or is the fallback when no split exists. Without that gate, rare
    whole tokens become free wildcards and the shorter derivation always wins."""
    inventory = sorted(pieces | set(tokens))
    index = {p: i for i, p in enumerate(inventory)}
    starts = [0]; lengths = []; first = []; second = []
    for t in tokens:
        splits = [i for i in range(1, len(t)) if t[:i] in pieces and t[i:] in pieces]
        if t in pieces or not splits:
            lengths.append(1); first.append(index[t]); second.append(-1)
        for i in splits:
            lengths.append(2); first.append(index[t[:i]]); second.append(index[t[i:]])
        starts.append(len(lengths))
    return inventory, (np.array(starts, dtype=np.int64), np.array(lengths, dtype=np.int64),
                       np.array(first, dtype=np.int64), np.array(second, dtype=np.int64))


def joint_em(tokens, prior, minimum=3, restarts=4, iterations=60, smoothing=.1, seed=0, cap=1800, warm_start=None, warm_mass=.9):
    """Run EM over latent parses and emissions; return the best restart's decode.

    `warm_start` is a role-unit key ('u:piece' -> letter); when given, the first restart
    initializes each known unit's emission with `warm_mass` on that letter instead of a
    random table, so EM re-estimates parses around an already refined key.
    """
    started = time.monotonic()
    base = prior.base
    pieces = candidate_pieces(tokens, minimum)
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
            best = dict(recovered=''.join(recovered), segmentation=chosen, log_likelihood=float(loglik), best_restart=r,
                        pieces=len(inventory), parses=int(len(lengths)))
    best.update(restarts=len(history), restart_scores=history, iterations=iterations, minimum=minimum,
                seconds=time.monotonic() - started, cap_hit=time.monotonic() - started > cap)
    return best


def resegment(tokens, pieces, mapping, prior, unknown_bits=None):
    """Greedy left-to-right parse choice under a fixed role-unit key and the character prior.

    Each token takes the parse whose letters cost least given the letters decoded so far.
    Units missing from `mapping` cost `unknown_bits` per letter (default: a uniform letter plus a new key entry).
    Returns the new segmentation and the letters it implies.
    """
    base, order = prior.base, prior.order
    if unknown_bits is None:
        # a letter at uniform cost plus a new one-letter key entry, as in the beam's key code
        unknown_bits = float(np.log2(base)) + 1. + float(np.ceil(np.log2(base)))
    lookup = {c: i for i, c in enumerate(prior.alphabet)}
    history = []
    segmentation = []; recovered = []

    def letter_cost(letter, context):
        n = min(len(context), order - 1)
        index = lookup[letter]; power = base
        for k in range(1, n + 1):
            index += lookup[context[-k]] * power; power *= base
        return float(prior.logs[n][index])

    for t in tokens:
        options = []
        splits = [i for i in range(1, len(t)) if t[:i] in pieces and t[i:] in pieces]
        if t in pieces or not splits:
            options.append((('u:' + t,), (t,)))
        for i in splits:
            options.append((('p:' + t[:i], 's:' + t[i:]), (t[:i], t[i:])))
        best, best_cost, best_letters = None, None, None
        for units, parts in options:
            cost = 0.; context = list(history); letters = []
            for u in units:
                if u in mapping:
                    letter = mapping[u]; cost += letter_cost(letter, context)
                else:
                    letter = None; cost += unknown_bits
                letters.append(letter)
                if letter is not None:
                    context.append(letter)
            if best_cost is None or cost < best_cost:
                best, best_cost, best_letters = parts, cost, letters
        segmentation.append(best)
        for letter in best_letters:
            if letter is not None:
                history.append(letter); recovered.append(letter)
            else:
                recovered.append('?')
    return dict(segmentation=segmentation, recovered=''.join(recovered))
