"""Bigram HMM decipherment: fixed syllable transitions, emissions learned by EM (Knight et al. 2006)."""

import numpy as np
from numba import njit


def bigram_model(stream, alphabet, smoothing=0.5):
    index = {s: i for i, s in enumerate(alphabet)}
    k = len(alphabet)
    counts = np.full((k, k), smoothing)
    unigram = np.full(k, smoothing)
    ids = [index[s] for s in stream]
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
    for a in ids:
        unigram[a] += 1
    return unigram / unigram.sum(), counts / counts.sum(axis=1, keepdims=True)


@njit(cache=True)
def _forward_backward(obs, start, trans, emit):
    t_len, k = len(obs), len(start)
    alpha = np.zeros((t_len, k))
    scale = np.zeros(t_len)
    for j in range(k):
        alpha[0, j] = start[j] * emit[j, obs[0]]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    for t in range(1, t_len):
        for j in range(k):
            s = 0.0
            for i in range(k):
                s += alpha[t - 1, i] * trans[i, j]
            alpha[t, j] = s * emit[j, obs[t]]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]
    beta = np.ones(k)
    counts = np.zeros(emit.shape)
    for j in range(k):
        counts[j, obs[t_len - 1]] += alpha[t_len - 1, j]
    for t in range(t_len - 2, -1, -1):
        new = np.zeros(k)
        for i in range(k):
            s = 0.0
            for j in range(k):
                s += trans[i, j] * emit[j, obs[t + 1]] * beta[j]
            new[i] = s / scale[t + 1]
        beta = new
        norm = 0.0
        for i in range(k):
            norm += alpha[t, i] * beta[i]
        for i in range(k):
            counts[i, obs[t]] += alpha[t, i] * beta[i] / norm
    return counts, np.log(scale).sum()


@njit(cache=True)
def _viterbi(obs, log_start, log_trans, log_emit):
    t_len, k = len(obs), len(log_start)
    score = log_start + log_emit[:, obs[0]]
    back = np.zeros((t_len, k), dtype=np.int64)
    for t in range(1, t_len):
        new = np.empty(k)
        for j in range(k):
            best, arg = -1e300, 0
            for i in range(k):
                v = score[i] + log_trans[i, j]
                if v > best:
                    best, arg = v, i
            new[j] = best + log_emit[j, obs[t]]
            back[t, j] = arg
        score = new
    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = np.argmax(score)
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decipher(signs, start, trans, restarts=20, iterations=200, seed=0, smoothing=0.1):
    """Returns (state index per token, log likelihood) of the best restart."""
    inventory = sorted(set(signs))
    lookup = {s: i for i, s in enumerate(inventory)}
    obs = np.array([lookup[s] for s in signs], dtype=np.int64)
    k, m = len(start), len(inventory)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        emit = rng.uniform(size=(k, m))
        emit /= emit.sum(axis=1, keepdims=True)
        for _ in range(iterations):
            counts, ll = _forward_backward(obs, start, trans, emit)
            emit = counts + smoothing
            emit /= emit.sum(axis=1, keepdims=True)
        counts, ll = _forward_backward(obs, start, trans, emit)
        if best is None or ll > best[1]:
            best = (emit, ll)
    path = _viterbi(obs, np.log(start), np.log(trans), np.log(best[0]))
    return path, best[1]
