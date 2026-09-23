"""Context agreement: does a word's matched meaning predict where it sits on the tablet?

For language L, each sample word within ``theta`` of an L form inherits that form's class:
*name* (proper noun) or *common*. The statistic counts matched words whose class agrees with
their tablet context, where *name* agrees with ``entry`` and *common* with every other label:

    A_L = #{matched w : is_name(w) == (label(w) == "entry")}

The null permutes context labels among all sample words of the same syllable length, so word
length (which affects both matching and position) cannot create agreement. ``z`` is
``(A_L - mean) / max(sd, 1)`` over the permutations.
"""

import numpy as np


def name_flags(lexicon, rule):
    """``{form: bool}``; ``rule`` is ``any`` (some entry is a name) or ``majority``."""
    if rule == "any":
        return {form: r["name_share"] > 0 for form, r in lexicon.items()}
    if rule == "majority":
        return {form: r["name_share"] > 0.5 for form, r in lexicon.items()}
    raise ValueError(rule)


def strata(words):
    lengths = np.array([len(w) for w in words])
    return [np.flatnonzero(lengths == n) for n in np.unique(lengths)]


def permuted_labels(entry, groups, rng, repeats):
    """``repeats`` copies of ``entry`` with values shuffled inside each length group."""
    out = np.repeat(entry[None, :], repeats, axis=0)
    for group in groups:
        if len(group) < 2:
            continue
        block = out[:, group]
        out[:, group] = rng.permuted(block, axis=1)
    return out


def context_scores(matcher, flags, words, labels, theta, rng, repeats=1000):
    """``{language: {matched, agree, null_mean, null_sd, z}}`` for one sample.

    ``flags[language]`` maps each lexicon form to its name flag; ``labels`` aligns with ``words``.
    """
    words = list(words)
    entry = np.array([label == "entry" for label in labels])
    perms = permuted_labels(entry, strata(words), rng, repeats)
    scores = {}
    for name in matcher.names:
        nearest = matcher.nearest(words, name)
        forms = matcher.forms[name]
        matched = np.array([d <= theta for d, _ in nearest])
        is_name = np.array([flags[name][forms[i]] if d <= theta else False for d, i in nearest])
        agree = int((matched & (is_name == entry)).sum())
        null = (matched[None, :] & (is_name[None, :] == perms)).sum(axis=1).astype(float)
        mean, sd = null.mean(), null.std(ddof=1)
        scores[name] = {
            "matched": int(matched.sum()),
            "matched_names": int((matched & is_name).sum()),
            "agree": agree,
            "null_mean": float(mean),
            "null_sd": float(sd),
            "z": float((agree - mean) / max(sd, 1.0)),
        }
    return scores
