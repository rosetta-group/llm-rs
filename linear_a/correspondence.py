"""Literal-reading correspondence audit; historical normalisation stays frozen."""

import collections
import re

import numpy as np

from linear_a import contexts


# Literal sign labels: q and k, a and a3, ta and ta2 remain distinct.
WORD = re.compile(r'[a-z]+[0-9]*(?:-[a-z]+[0-9]*){2,}')
OPEN = {'[': ']', '⟦': '⟧', '<': '>', '{': '}'}


def literal(text):
    return tuple(text.split('-')) if WORD.fullmatch(text) else None


def b_tokens(item):
    """Whole unmarked DĀMOS tokens, never substrings or bracket-stripped restorations.

    An open editorial bracket also excludes subsequent tokens until its close on that line.
    Broken tablet edges are marked independently on each transcription line: a trailing
    '[' in .A must not invalidate preserved words at the beginning of .B.
    A closing bracket or underdot on a token excludes that token too.
    """
    for line_n, line in enumerate((item.get('content') or '').splitlines(), 1):
        stack = []
        for raw in re.split(r'[\s,/]+', line):
            enclosed = bool(stack)
            for ch in raw:
                if ch in OPEN:
                    stack.append(OPEN[ch])
                elif ch in OPEN.values() and stack and stack[-1] == ch:
                    stack.pop()
            word = literal(raw)
            if word and not enclosed:
                yield {'word': word, 'raw': raw, 'line_number': line_n, 'line': line}


def a_tokens(corpus):
    """Require agreement of word, sign-reading/certainty and intact Unicode-run layers.

    Use the historical fixed 80% logogram boundary rule. If multiple source sign spans have
    the same IDs, require ALL of them to be certain syllabograms with the same readings;
    this avoids borrowing certainty from a different occurrence. Source layers are collated
    readings, not independent witnesses. Unknown or editorially marked readings are excluded.
    """
    table = contexts.sign_table(corpus)
    for t in [t for t in table if t]:
        for variant in 'ABC':
            table.setdefault(t + variant, table[t])
    for doc, record in sorted(corpus.items()):
        if record.get('word_source') != 'sigla':
            continue
        words = set(record.get('words') or [])
        signs = record.get('signs') or []
        conflicts = {c.get(side) for c in record.get('conflicts', [])
                     for side in ('ours', 'theirs') if isinstance(c.get(side), str)}
        for line_n, line in enumerate((record.get('unicode_text') or '').splitlines(), 1):
            runs, run, damaged = [], [], False

            def flush():
                nonlocal damaged
                if run and not damaged:
                    runs.append(tuple(run))
                run.clear()
                damaged = False

            for ch in line:
                t = contexts._sign_type(ch)
                if contexts._is_numeral(ch) or contexts._is_fraction(ch):
                    flush()
                elif t:
                    if table.get(t, (None, 0))[1] >= .8:
                        flush()
                    else:
                        run.append(t)
                elif ch.isspace() or ch in '\U00010100\U00010101|':
                    flush()
                else:
                    damaged = True
            flush()
            for ids in runs:
                reading = '-'.join(table.get(t, (None, 0))[0] or '*' for t in ids)
                word = literal(reading)
                if not word or reading not in words or reading in conflicts:
                    continue
                spans = [signs[i:i+len(ids)] for i in range(len(signs)-len(ids)+1)
                         if tuple(s.get('type') for s in signs[i:i+len(ids)]) == ids]
                if not spans or not all(all(s.get('certain') is True
                        and s.get('role') == 'syllabogram' and s.get('reading') == r
                        for s, r in zip(span, word)) for span in spans):
                    continue
                yield {'word': word, 'document': doc, 'line_number': line_n, 'line': line,
                       'sign_ids': ids, 'sign_numbers': [[s['n'] for s in span] for span in spans],
                       'parent_object': record.get('parent_object') or doc}


def pairs(a_words, b_words):
    b = set(b_words)
    return sorted({(w, w[:-1] + ('ro',)) for w in a_words
                   if len(w) >= 3 and w[-1] in ('re', 'ru') and w[:-1] + ('ro',) in b})


def score(a_words, b_words):
    # re and ru variants of the same stem get one vote, as do duplicate synthetic words.
    return len({a[:-1] for a, _ in pairs(a_words, b_words)})


def onset(sign):
    """Literal final onset: re/ru/ro/ra2 -> r; qe -> q; vowels -> empty."""
    return re.sub(r'[aeiou][0-9]*$', '', sign)


def groups(words, mode):
    if mode not in ('length', 'length_onset'):
        raise ValueError(mode)
    result = collections.defaultdict(list)
    for i, w in enumerate(words):
        result[(len(w), onset(w[-1]) if mode == 'length_onset' else '')].append(i)
    return list(result.values())


def permute_endings(words, buckets, rng):
    out = list(words)
    for ids in buckets:
        endings = rng.permutation([words[i][-1] for i in ids])
        for i, end in zip(ids, endings):
            out[i] = words[i][:-1] + (str(end),)
    return out


def null_scores(a_words, b_words, mode, repeats, seed):
    """Condition on real stems and observed final-sign inventories, with B fixed."""
    words = sorted(set(a_words))
    buckets = groups(words, mode)
    rng = np.random.default_rng(seed)
    # Only stems ending in B ro can score. Evaluate their assignments without rebuilding
    # the whole corpus; random permutations still involve every slot in each bucket.
    eligible = {w[:-1] for w in b_words if w[-1] == 'ro'}
    finals = np.array([w[-1] for w in words])
    relevant = [(ids, [(j, words[i][:-1]) for j, i in enumerate(ids)
                       if words[i][:-1] in eligible]) for ids in buckets]
    values = []
    for _ in range(repeats):
        hit = set()
        for ids, targets in relevant:
            ends = rng.permutation(finals[ids])
            hit.update(stem for j, stem in targets if ends[j] in ('re', 'ru'))
        values.append(len(hit))
    mutable = [ids for ids in buckets if len({words[i][-1] for i in ids}) > 1]
    return values, {'buckets': len(buckets), 'mutable_buckets': len(mutable),
                    'mutable_slots': sum(map(len, mutable)),
                    'eligible_stems': len({w[:-1] for w in words if w[:-1] in eligible})}
