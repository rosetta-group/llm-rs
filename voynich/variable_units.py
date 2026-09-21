"""Key beam search over units that may each stand for one or two letters.

Two additions to the one-letter key beam in :mod:`voynich.homophonic`:

1. Every unit may map to a single letter or to one of a fixed list of
   plaintext bigrams, so the model class covers Naibbe-style one/two-letter
   units and the artificial variable-length control.
2. The partial score is a description length, not bits per recovered
   letter: letters under the gap-aware character prior, plus explicit key
   bits per mapped unit, plus the uniform-code cost of choosing among units
   that share the same expansion. The last term charges collapsing many
   units onto one letter, which a bits-per-letter score rewards for free.

A second helper induces sub-token pieces from space-delimited tokens with a
two-part code, for ciphers whose visible tokens concatenate up to two units.
Language data, widths and CPU caps belong to the experiment protocol.
"""
from collections import Counter
import math
import time

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def variable_scores(first, second, values, freq, costs, offsets, base, order, single_bits, double_bits):
    hypotheses, size = first.shape
    n_exp = base + base * base
    out = np.zeros(hypotheses)
    for h in prange(hypotheses):
        total = 0.
        hist = np.zeros(order, dtype=np.int64)
        run = 0
        for i in range(len(values)):
            u = values[i]
            a = first[h, u]
            if a < 0:
                run = 0
                continue
            n = min(run, order - 1)
            index = a
            power = base
            for k in range(n):
                index += hist[k] * power
                power *= base
            total += costs[offsets[n] + index]
            for k in range(order - 2, 0, -1):
                hist[k] = hist[k - 1]
            hist[0] = a
            run += 1
            b = second[h, u]
            if b >= 0:
                n = min(run, order - 1)
                index = b
                power = base
                for k in range(n):
                    index += hist[k] * power
                    power *= base
                total += costs[offsets[n] + index]
                for k in range(order - 2, 0, -1):
                    hist[k] = hist[k - 1]
                hist[0] = b
                run += 1
        units_per = np.zeros(n_exp, dtype=np.int64)
        occ_per = np.zeros(n_exp, dtype=np.int64)
        for s in range(size):
            a = first[h, s]
            if a < 0:
                continue
            b = second[h, s]
            if b < 0:
                e = a
                total += single_bits
            else:
                e = base + a * base + b
                total += double_bits
            units_per[e] += 1
            occ_per[e] += freq[s]
        for e in range(n_exp):
            if units_per[e] > 1:
                total += occ_per[e] * np.log2(units_per[e])
        out[h] = total
    return out


def bigram_options(prior, count):
    """The `count` most probable plaintext bigrams under the prior's own statistics."""
    base = prior.base
    joint = (prior.probabilities[0][:, None] * prior.probabilities[1].reshape(base, base)).ravel()
    order = np.argsort(-joint, kind='stable')[:count]
    return [(int(i // base), int(i % base)) for i in order]


def variable_beam(units, prior, bigrams=(), width=512, cap=900, max_homophones=0):
    """Frequency-ordered key beam; each unit takes one letter or a listed bigram.

    `max_homophones` > 0 also applies a hard cap on units sharing one expansion.
    Returns the best complete key, the recovered letters and the partial score.
    """
    start = time.monotonic()
    base, order = prior.base, prior.order
    counts = Counter(units)
    inventory = sorted(counts, key=lambda s: (-counts[s], s))
    lookup = {s: i for i, s in enumerate(inventory)}
    values = np.array([lookup[s] for s in units], dtype=np.int32)
    freq = np.array([counts[s] for s in inventory], dtype=np.int64)
    letters = [(a, -1) for a in range(base)] + [(a, b) for a, b in bigrams]
    opt_a = np.array([a for a, _ in letters], dtype=np.int16)
    opt_b = np.array([b for _, b in letters], dtype=np.int16)
    costs = np.concatenate(prior.logs)
    offsets = np.array([sum(len(p) for p in prior.logs[:n]) for n in range(order)], dtype=np.int64)
    letter_bits = math.ceil(math.log2(base))
    first = np.full((1, len(inventory)), -1, dtype=np.int16)
    second = np.full((1, len(inventory)), -1, dtype=np.int16)
    scores = np.zeros(1)
    for depth, symbol in enumerate(range(len(inventory))):
        h = len(first)
        ext_first = np.repeat(first, len(letters), axis=0)
        ext_second = np.repeat(second, len(letters), axis=0)
        ext_first[:, symbol] = np.tile(opt_a, h)
        ext_second[:, symbol] = np.tile(opt_b, h)
        if max_homophones > 0:
            same = (ext_first == ext_first[:, symbol][:, None]) & (ext_second == ext_second[:, symbol][:, None])
            allowed = same.sum(axis=1) <= max_homophones
            ext_first, ext_second = ext_first[allowed], ext_second[allowed]
        scores = variable_scores(ext_first, ext_second, values, freq, costs, offsets, base, order,
                                 1. + letter_bits, 1. + 2 * letter_bits)
        keep = np.argsort(scores, kind='stable')[:width]
        first, second, scores = ext_first[keep], ext_second[keep], scores[keep]
        if time.monotonic() - start > cap and depth + 1 < len(inventory):
            return dict(status='budget_exhausted', mapped=depth + 1, inventory=len(inventory),
                        seconds=time.monotonic() - start, width=width)
    mapping = {}
    for s, a, b in zip(inventory, first[0], second[0]):
        mapping[s] = prior.alphabet[a] + (prior.alphabet[b] if b >= 0 else '')
    return dict(status='complete', mapping=mapping, recovered=''.join(mapping[u] for u in units),
                search_bits=float(scores[0]), seconds=time.monotonic() - start, width=width,
                bigram_options=len(bigrams), max_homophones=max_homophones, inventory=len(inventory),
                two_letter_units=int((second[0] >= 0).sum()))


def string_bits(piece):
    data = piece.encode('utf8')
    return 2 * int(math.log2(len(data) + 1)) + 1 + 8 * len(data)


def induce_pieces(tokens, passes=8, max_pieces=2, seed=0, roles=False):
    """Two-part-code segmentation of each token type into at most `max_pieces` pieces.

    Starts from whole tokens and greedily re-segments each type when the split
    lowers total bits: piece occurrences under their empirical distribution plus
    the literal cost of the piece inventory. With `roles`, pieces are counted in
    three positional distributions (whole token, first part, second part), which
    models codebooks whose tables differ by position. Deterministic given the seed.
    """
    if max_pieces != 2:
        raise ValueError('Only whole tokens or one split are implemented')
    counts = Counter(tokens)
    segmentation = {t: (t,) for t in counts}
    role_of = (lambda parts, i: ('u' if len(parts) == 1 else ('p', 's')[i]) if roles else 'x')
    piece_counts = Counter()  # keyed by (role, piece)
    for t, n in counts.items():
        piece_counts[(role_of((t,), 0), t)] += n
    total = sum(piece_counts.values())

    def cost(parts, occurrences, delta_types):
        bits = 0.
        local = Counter((role_of(parts, i), p) for i, p in enumerate(parts))
        for key, k in local.items():
            c = piece_counts.get(key, 0) + k * occurrences
            bits += -k * occurrences * math.log2(c / (total + delta_types))
        return bits

    rng = np.random.default_rng(seed)
    changed = True
    for _ in range(passes):
        if not changed:
            break
        changed = False
        for t in rng.permutation(sorted(counts)):
            t = str(t)
            n = counts[t]
            current = segmentation[t]
            for i, piece in enumerate(current):
                key = (role_of(current, i), piece)
                piece_counts[key] -= n
                if piece_counts[key] == 0:
                    del piece_counts[key]
            total = sum(piece_counts.values())
            options = [(t,)] + [(t[:i], t[i:]) for i in range(1, len(t))]
            best, best_bits = None, None
            strings = {p for _, p in piece_counts}
            for option in options:
                new_types = [p for p in set(option) if p not in strings]
                bits = cost(option, n, n * len(option)) + sum(string_bits(p) for p in new_types)
                if best_bits is None or bits < best_bits - 1e-9:
                    best, best_bits = option, bits
            if best != current:
                changed = True
            segmentation[t] = best
            for i, piece in enumerate(best):
                piece_counts[(role_of(best, i), piece)] += n
            total = sum(piece_counts.values())
    pieces = [p for t in tokens for p in segmentation[t]]
    strings = {p for _, p in piece_counts}
    inventory_bits = sum(string_bits(p) for p in strings)
    sequence_bits = sum(-n * math.log2(n / total) for n in piece_counts.values())
    return dict(pieces=pieces, segmentation=segmentation, piece_types=len(strings),
                token_types=len(counts), total_bits=inventory_bits + sequence_bits,
                inventory_bits=inventory_bits, sequence_bits=sequence_bits,
                split_types=sum(len(v) > 1 for v in segmentation.values()), roles=roles)


def _prepare(units, prior):
    counts = Counter(units)
    inventory = sorted(counts, key=lambda s: (-counts[s], s))
    lookup = {s: i for i, s in enumerate(inventory)}
    values = np.array([lookup[s] for s in units], dtype=np.int32)
    freq = np.array([counts[s] for s in inventory], dtype=np.int64)
    costs = np.concatenate(prior.logs)
    offsets = np.array([sum(len(p) for p in prior.logs[:n]) for n in range(prior.order)], dtype=np.int64)
    letter_bits = math.ceil(math.log2(prior.base))
    return inventory, values, freq, costs, offsets, 1. + letter_bits, 1. + 2 * letter_bits


def key_arrays(inventory, mapping, prior):
    first = np.array([[prior.alphabet.index(mapping[s][0]) for s in inventory]], dtype=np.int16)
    second = np.array([[prior.alphabet.index(mapping[s][1]) if len(mapping[s]) > 1 else -1 for s in inventory]], dtype=np.int16)
    return first, second


def objective(units, prior, mapping):
    """The beam's partial description length for a complete key."""
    inventory, values, freq, costs, offsets, sb, db = _prepare(units, prior)
    first, second = key_arrays(inventory, mapping, prior)
    return float(variable_scores(first, second, values, freq, costs, offsets, prior.base, prior.order, sb, db)[0])


def refine(units, prior, mapping, bigrams=(), sweeps=50, kicks=20, kick_size=6, seed=0, cap=300):
    """Iterated local search on a complete key under the same objective as the beam.

    Each sweep tries every expansion for every unit and keeps improvements, then a
    batch of pairwise swaps. When a sweep finds nothing, a random kick perturbs
    `kick_size` units and the descent restarts; the best key seen is returned.
    """
    start = time.monotonic()
    inventory, values, freq, costs, offsets, sb, db = _prepare(units, prior)
    base, order = prior.base, prior.order
    letters = [(a, -1) for a in range(base)] + [(a, b) for a, b in bigrams]
    opt_a = np.array([a for a, _ in letters], dtype=np.int16)
    opt_b = np.array([b for _, b in letters], dtype=np.int16)
    first, second = key_arrays(inventory, mapping, prior)
    score = lambda f, s: variable_scores(f, s, values, freq, costs, offsets, base, order, sb, db)
    current = float(score(first, second)[0])
    best_first, best_second, best = first.copy(), second.copy(), current
    rng = np.random.default_rng(seed)
    size = len(inventory)
    pairs = np.array([(i, j) for i in range(size) for j in range(i + 1, size)], dtype=np.int64)
    kicks_done = 0; evaluations = 0; improvements = 0
    while True:
        improved = False
        for s in rng.permutation(size):
            f = np.repeat(first, len(letters), axis=0); g = np.repeat(second, len(letters), axis=0)
            f[:, s] = opt_a; g[:, s] = opt_b
            sc = score(f, g); evaluations += len(sc); k = int(np.argmin(sc))
            if sc[k] < current - 1e-9:
                first, second, current = f[k:k + 1].copy(), g[k:k + 1].copy(), float(sc[k]); improved = True; improvements += 1
            if time.monotonic() - start > cap:
                break
        if time.monotonic() - start <= cap and len(pairs):
            f = np.repeat(first, len(pairs), axis=0); g = np.repeat(second, len(pairs), axis=0)
            rows = np.arange(len(pairs))
            f[rows, pairs[:, 0]], f[rows, pairs[:, 1]] = first[0, pairs[:, 1]], first[0, pairs[:, 0]]
            g[rows, pairs[:, 0]], g[rows, pairs[:, 1]] = second[0, pairs[:, 1]], second[0, pairs[:, 0]]
            sc = score(f, g); evaluations += len(sc); k = int(np.argmin(sc))
            if sc[k] < current - 1e-9:
                first, second, current = f[k:k + 1].copy(), g[k:k + 1].copy(), float(sc[k]); improved = True; improvements += 1
        if current < best - 1e-9:
            best_first, best_second, best = first.copy(), second.copy(), current
        if time.monotonic() - start > cap or (not improved and kicks_done >= kicks):
            break
        if not improved:
            kicks_done += 1
            first, second = best_first.copy(), best_second.copy()
            for s in rng.choice(size, size=min(kick_size, size), replace=False):
                k = rng.integers(len(letters)); first[0, s] = opt_a[k]; second[0, s] = opt_b[k]
            current = float(score(first, second)[0])
    result_mapping = {s: prior.alphabet[a] + (prior.alphabet[b] if b >= 0 else '')
                      for s, a, b in zip(inventory, best_first[0], best_second[0])}
    return dict(status='complete', mapping=result_mapping, recovered=''.join(result_mapping[u] for u in units),
                search_bits=best, seconds=time.monotonic() - start, kicks=kicks_done,
                evaluations=int(evaluations), improvements=improvements, cap_hit=time.monotonic() - start > cap)


@njit(cache=True)
def _anneal_core(values, key, positions, starts, freq, costs, offsets, base, order,
                 proposals, t0, t1, seed, stamp):
    """Metropolis search over one-letter keys with incremental n-gram deltas.

    `positions`/`starts` is a CSR list of occurrence positions per unit. The state
    cost is the full-context prior cost of the plaintext plus the uniform-code cost of
    choosing among units that share a letter. Returns the best key visited.
    """
    np.random.seed(seed)
    n = len(values); size = len(key)
    text = np.empty(n, dtype=np.int64)
    for i in range(n):
        text[i] = key[values[i]]
    units_per = np.zeros(base, dtype=np.int64); occ_per = np.zeros(base, dtype=np.int64)
    for s in range(size):
        units_per[key[s]] += 1; occ_per[key[s]] += freq[s]

    def window_cost(text, end):
        # cost of the letter at `end` given up to order-1 previous letters
        m = min(end, order - 1)
        index = text[end]; power = base
        for k in range(1, m + 1):
            index += text[end - k] * power; power *= base
        return costs[offsets[m] + index]

    current = 0.
    for i in range(n):
        current += window_cost(text, i)
    for e in range(base):
        if units_per[e] > 1:
            current += occ_per[e] * np.log2(units_per[e])
    best = current; best_key = key.copy()
    affected = np.empty(n, dtype=np.int64)
    ratio = (t1 / t0) ** (1. / max(1, proposals - 1))
    temperature = t0
    accepted = 0
    for step in range(proposals):
        u = np.random.randint(size)
        old = key[u]; new = np.random.randint(base - 1)
        if new >= old:
            new += 1
        # collect affected window ends without duplicates
        count = 0; mark = step + 1
        for q in range(starts[u], starts[u + 1]):
            p = positions[q]
            for e in range(p, min(n, p + order)):
                if stamp[e] != mark:
                    stamp[e] = mark; affected[count] = e; count += 1
        delta = 0.
        for k in range(count):
            delta -= window_cost(text, affected[k])
        for q in range(starts[u], starts[u + 1]):
            text[positions[q]] = new
        for k in range(count):
            delta += window_cost(text, affected[k])
        # residual change
        f = freq[u]
        if units_per[old] > 1:
            delta -= occ_per[old] * np.log2(units_per[old])
        if units_per[old] > 2:
            delta += (occ_per[old] - f) * np.log2(units_per[old] - 1)
        if units_per[new] > 1:
            delta -= occ_per[new] * np.log2(units_per[new])
        delta += (occ_per[new] + f) * np.log2(units_per[new] + 1)
        if delta <= 0 or np.random.random() < np.exp(-delta / temperature):
            key[u] = new; current += delta; accepted += 1
            units_per[old] -= 1; occ_per[old] -= f; units_per[new] += 1; occ_per[new] += f
            if current < best - 1e-9:
                best = current; best_key[:] = key
        else:
            for q in range(starts[u], starts[u + 1]):
                text[positions[q]] = old
        temperature *= ratio
    return best_key, best, accepted


def anneal(units, prior, mapping=None, proposals=2_000_000, t0=3., t1=.05, seed=0, restarts=1, cap=600):
    """Simulated annealing over complete one-letter keys; restarts keep the best result."""
    start = time.monotonic()
    inventory, values, freq, costs, offsets, single_bits, _ = _prepare(units, prior)
    base, order = prior.base, prior.order
    order_idx = np.argsort(values, kind='stable')
    positions = order_idx.astype(np.int64)
    starts = np.zeros(len(inventory) + 1, dtype=np.int64)
    np.cumsum(freq, out=starts[1:])
    rng = np.random.default_rng(seed)
    best_key, best, history = None, None, []
    stamp = np.zeros(len(values), dtype=np.int64)
    for r in range(restarts):
        if r and time.monotonic() - start > cap:
            break
        if mapping is not None and r == 0:
            key = np.array([prior.alphabet.index(mapping[s][0]) for s in inventory], dtype=np.int64)
        else:
            key = rng.integers(base, size=len(inventory)).astype(np.int64)
        stamp[:] = 0
        k, score, accepted = _anneal_core(values, key, positions, starts, freq, costs, offsets, base, order,
                                          proposals, t0, t1, int(rng.integers(2**31)), stamp)
        history.append(float(score))
        if best is None or score < best:
            best, best_key = float(score), k.copy()
    result = {s: prior.alphabet[c] for s, c in zip(inventory, best_key)}
    return dict(status='complete', mapping=result, recovered=''.join(result[u] for u in units),
                anneal_bits=best + single_bits * len(inventory), restart_scores=history,
                seconds=time.monotonic() - start, proposals=proposals, restarts=len(history))
