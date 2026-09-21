"""Nuhn et al. key beam search with their 2014 longest-known-context score.

Independent implementation, not the authors' UNRAVEL package. The extension
order uses their 2014 order beam for inventories up to 64 symbols, otherwise
the 2013 frequency order. Language and CPU settings belong to the protocol.
"""
from collections import Counter
from functools import lru_cache
import time

import numpy as np
from numba import njit


@njit(cache=True)
def partial_scores(keys, values, costs, offsets, base, order):
    result = np.zeros(len(keys))
    for h in range(len(keys)):
        score = 0.
        for i in range(len(values)):
            current = keys[h, values[i]]
            if current < 0:
                continue
            index = current
            power = base
            n = 1
            while n < order and i-n >= 0:
                previous = keys[h, values[i-n]]
                if previous < 0:
                    break
                index += previous * power
                power *= base
                n += 1
            score += costs[offsets[n-1] + index]
        result[h] = score
    return result


@njit(cache=True)
def order_scores(masks, values, weights):
    scores = np.zeros(len(masks))
    for j in range(len(masks)):
        run = 0
        for value in values:
            if masks[j] & (np.uint64(1) << np.uint64(value)):
                run = min(run+1,len(weights)-1)
                scores[j] += weights[run]
            else:
                run = 0
    return scores


@lru_cache(maxsize=32)
def extension_order(sequence, size, lm_order, width=100):
    """2014 order search, deduplicating subsets with identical future options.

    Scores at the latest depth take priority; ties use the previous depth's score.
    Shorter order-5 weights adapt the paper's increasing emphasis on long contexts.
    """
    values = np.array(sequence,dtype=np.int32)
    if size > 64:
        counts = Counter(sequence)
        return tuple(sorted(range(size),key=lambda i:(-counts[i],i)))
    weights = np.array([0.,0.,1.,1.,2.,3.][:lm_order+1])
    if lm_order > 5:
        raise ValueError('Specify extension weights for higher language-model order')
    states = [(0, (), ())]
    for depth in range(size):
        expanded = {}
        for mask, order, history in states:
            for symbol in range(size):
                if mask & (1 << symbol):
                    continue
                target = mask | (1 << symbol)
                candidate = (order+(symbol,),history)
                if target not in expanded or history > expanded[target][1]:
                    expanded[target] = candidate
        masks = list(expanded)
        scores = order_scores(np.array(masks,dtype=np.uint64),values,weights)
        states = [(mask,expanded[mask][0],(float(score),)+expanded[mask][1]) for mask,score in zip(masks,scores)]
        states.sort(key=lambda r:r[2],reverse=True)
        states = states[:width]
    return states[0][1]


def beam_search(units, prior, width=256, max_homophones=2, cap=120):
    start = time.monotonic()
    inventory = sorted(set(units))
    lookup = {s:i for i,s in enumerate(inventory)}
    values = np.array([lookup[s] for s in units], dtype=np.int32)
    order = extension_order(tuple(values),len(inventory),prior.order)
    if max_homophones * prior.base < len(inventory):
        return dict(status='capacity_exceeded', inventory=len(inventory), seconds=time.monotonic()-start)
    costs = np.concatenate(prior.logs)
    offsets = np.array([sum(len(p) for p in prior.logs[:n]) for n in range(prior.order)], dtype=np.int64)
    keys = np.full((1,len(inventory)), -1, dtype=np.int16)
    for depth, symbol in enumerate(order):
        extended = np.repeat(keys, prior.base, axis=0)
        letters = np.tile(np.arange(prior.base), len(keys))
        allowed = (extended == letters[:,None]).sum(axis=1) < max_homophones
        extended[:,symbol] = letters
        extended = extended[allowed]
        scores = partial_scores(extended, values, costs, offsets, prior.base, prior.order)
        keep = np.argsort(scores, kind='stable')[:width]
        keys = extended[keep]
        if time.monotonic()-start > cap and depth+1 < len(order):
            return dict(status='budget_exhausted', mapped=depth+1, inventory=len(inventory), seconds=time.monotonic()-start)
    best = keys[0]
    return dict(status='complete', mapping={s:prior.alphabet[c] for s,c in zip(inventory,best)},
                recovered=''.join(prior.alphabet[c] for c in best[values]),
                search_bits=float(scores[keep[0]]), seconds=time.monotonic()-start,
                width=width, max_homophones=max_homophones,
                extension_order='2014-beam-100' if len(inventory)<=64 else '2013-frequency')


@njit(cache=True)
def expectation(values, emission, unigram, bigram, trigram):
    """Scaled forward/backward over pairs of plaintext letters."""
    length, base = len(values), len(unigram)
    forward = np.zeros((length, base, base))
    scales = np.zeros(length)
    first = unigram * emission[:,values[0]]
    scales[0] = first.sum()
    first /= scales[0]
    for a in range(base):
        for b in range(base):
            forward[1,a,b] = first[a]*bigram[a,b]*emission[b,values[1]]
    scales[1] = forward[1].sum()
    forward[1] /= scales[1]
    for t in range(2,length):
        for b in range(base):
            for c in range(base):
                total = 0.
                for a in range(base):
                    total += forward[t-1,a,b]*trigram[a,b,c]
                forward[t,b,c] = total*emission[c,values[t]]
        scales[t] = forward[t].sum()
        forward[t] /= scales[t]
    counts = np.zeros_like(emission)
    posterior = np.zeros((length,base))
    backward = np.ones((base,base))
    for t in range(length-1,0,-1):
        joint = forward[t]*backward
        joint /= joint.sum()
        for b in range(base):
            posterior[t,b] = joint[:,b].sum()
            counts[b,values[t]] += posterior[t,b]
        if t == 1:
            for a in range(base):
                posterior[0,a] = joint[a,:].sum()
                counts[a,values[0]] += posterior[0,a]
        else:
            previous = np.zeros((base,base))
            for a in range(base):
                for b in range(base):
                    for c in range(base):
                        previous[a,b] += trigram[a,b,c]*emission[c,values[t]]*backward[b,c]
            backward = previous/scales[t]
    return counts, posterior, np.log(scales).sum()


def hmm_em(units, prior, restarts=8, iterations=200, seed=42, cap=180):
    """Berg-Kirkpatrick/Klein model; bounded CPU restart adaptation, not 1M reproduction."""
    started = time.monotonic()
    inventory = sorted(set(units))
    lookup = {s:i for i,s in enumerate(inventory)}
    values = np.array([lookup[s] for s in units], dtype=np.int32)
    if len(values) < 2:
        raise ValueError('HMM requires two observations')
    base = prior.base
    unigram = prior.probabilities[0]
    bigram = (prior.probabilities[1].reshape(base,base)+unigram)/2
    trigram = (prior.probabilities[2].reshape(base,base,base)
               +prior.probabilities[1].reshape(1,base,base)+unigram)/3
    rng = np.random.default_rng(seed)
    best, history = None, []
    for restart in range(restarts):
        if restart and time.monotonic()-started >= cap:
            break
        emission = rng.uniform(size=(base,len(inventory)))
        emission /= emission.sum(axis=1,keepdims=True)
        for iteration in range(iterations):
            counts, posterior, score = expectation(values,emission,unigram,bigram,trigram)
            emission = counts+.1
            emission /= emission.sum(axis=1,keepdims=True)
        _, posterior, score = expectation(values,emission,unigram,bigram,trigram)
        history.append(float(score))
        if best is None or score > best['log_likelihood']:
            best = dict(recovered=''.join(prior.alphabet[i] for i in posterior.argmax(axis=1)),
                        log_likelihood=float(score), best_restart=restart)
    best.update(restarts=len(history), iterations=iterations, restart_scores=history,
                seconds=time.monotonic()-started, cap_hit=len(history)<restarts)
    return best
