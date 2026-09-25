"""A sign-only test of whether word endings predict numeric-entry position."""

import collections
import hashlib
import math
import re

import numpy as np

from linear_a import contexts


def sign_tokens(corpus):
    """Keep unknown syllabic signs as IDs; discard damaged runs, not unknown readings.

    Labels use the old, explicit next-item rule. No sound value or proposed word meaning is a
    model feature. Logogram roles use the same fixed 80% corpus threshold as round two.
    """
    table = contexts.sign_table(corpus)
    for t in [t for t in table if t]:
        for variant in 'ABC':
            table.setdefault(t+variant, table[t])
    rows = []
    for doc, record in sorted(corpus.items()):
        first_line = True
        site = record.get('site') or doc.split()[0]
        for line in (record.get('unicode_text') or '').splitlines():
            items, run = [], []
            damaged = False

            def flush():
                nonlocal damaged
                if run or damaged:
                    if damaged:
                        items.append(('barrier', None))
                    elif len(run) == 1 and table.get(run[0], (None, 0))[1] >= .5:
                        items.append(('logogram', run[0]))
                    else:
                        items.append(('word', tuple(run)))
                run.clear()
                damaged = False

            for ch in line:
                t = contexts._sign_type(ch)
                if contexts._is_numeral(ch) or contexts._is_fraction(ch):
                    flush()
                    if not items or items[-1][0] != 'number':
                        items.append(('number', None))
                elif t:
                    if table.get(t, (None, 0))[1] >= .8:
                        flush()
                        items.append(('logogram', t))
                    else:
                        run.append(t)
                elif ch.isspace() or ch in '\U00010100\U00010101|':
                    flush()
                else:
                    # U+1076B gaps, dashes, editorial marks: not word boundaries.
                    damaged = True
            flush()
            for w, label in contexts.label_line_items(items, first_line):
                if len(w) >= 2:
                    rows.append((doc, w, int(label == 'entry'), site))
            if items:
                first_line = False
    return rows, {t for t, (reading, _) in table.items() if reading}


def document_group(doc):
    # Faces and fragments of a numbered tablet stay together, e.g. HT 1a / HT 1b.
    return re.sub(r'(?<=\d)[a-z].*$', '', doc).strip()


def is_test(doc):
    return int(hashlib.sha256(('entry-endings-v1:' + document_group(doc)).encode()).hexdigest(), 16) % 5 == 0


def collapse(rows):
    grouped = collections.defaultdict(list)
    for doc, w, y, site in rows:
        grouped[w].append((doc, y, site))
    out = []
    for w, entries in sorted(grouped.items()):
        labels = {e[1] for e in entries}
        # Ambiguous type roles are excluded rather than forcing a majority label.
        if len(labels) != 1:
            continue
        site = min(collections.Counter(e[2] for e in entries),
                   key=lambda s: (-sum(e[2] == s for e in entries), s))
        out.append({'word': w, 'label': entries[0][1], 'site': site,
                    'documents': sorted({e[0] for e in entries})})
    return out


def partition(rows):
    train = [r for r in rows if not is_test(r[0])]
    seen = {r[1] for r in train}
    test = [r for r in rows if is_test(r[0]) and r[1] not in seen]
    return collapse(train), collapse(test)


def features(word, endings=False):
    f = collections.Counter('sign:' + s for s in word)
    f['length:' + str(min(len(word), 6))] += 1
    if endings:
        f['final:' + word[-1]] += 1
        f['final2:' + '/'.join(word[-2:])] += 1
    return f


def predict(train, test, endings=False):
    """Laplace-smoothed multinomial naive Bayes with equal class priors."""
    counts = {0: collections.Counter(), 1: collections.Counter()}
    for row in train:
        counts[row['label']].update(features(row['word'], endings))
    vocabulary = set(counts[0]) | set(counts[1])
    # Reserve an unknown bucket; no feature or hyperparameter is selected using test labels.
    denominators = {y: sum(c.values())+len(vocabulary)+1 for y, c in counts.items()}
    predictions = []
    for row in test:
        f = features(row['word'], endings)
        scores = {y: sum(n*math.log((c.get(k, 0)+1)/denominators[y]) for k,n in f.items())
                  for y,c in counts.items()}
        predictions.append(int(scores[1] > scores[0]))
    return np.array(predictions)


def balanced_accuracy(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if len(set(labels.tolist())) != 2:
        raise ValueError('both classes are required for balanced accuracy')
    return float(np.mean([np.mean(predictions[labels == y] == y) for y in (0, 1)]))


def permuted_training(train, rng):
    """Shuffle role labels among types within site and length bucket."""
    out = [dict(r) for r in train]
    groups = collections.defaultdict(list)
    for i, row in enumerate(train):
        groups[row['site'], min(len(row['word']), 6)].append(i)
    for ids in groups.values():
        labels = rng.permutation([train[i]['label'] for i in ids])
        for i, y in zip(ids, labels):
            out[i]['label'] = int(y)
    return out


def shuffled_words(rows, rng):
    """Keep each word's sign counts and length but destroy the final positions."""
    return [{**r, 'word': tuple(r['word'][i] for i in rng.permutation(len(r['word'])))} for r in rows]
