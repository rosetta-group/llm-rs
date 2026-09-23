"""Add training-only word counts and transitions to an immutable word model."""
from collections import Counter
from voynich.decipher import normalize


def augment(model, poems, weight):
    if type(weight) is not int or weight < 1:
        raise ValueError('Expected a positive integer weight')
    counts = Counter(model['counts'])
    following = {word: Counter(next_words) for word, next_words in model['following'].items()}
    lexicon = set(model['lexicon'])
    ids = []
    for row in poems:
        if row['split'] != 'train':
            continue
        if row['id'] in ids:
            raise ValueError('Duplicate training poem')
        ids.append(row['id'])
        words = normalize(' '.join(row['lines'])).split()
        lexicon.update(words)
        for word, count in Counter(words).items(): counts[word] += weight * count
        for (a, b), count in Counter(zip(['<s>'] + words, words)).items():
            following.setdefault(a, Counter())[b] += weight * count
    if not ids: raise ValueError('No training poems')
    return dict(counts=dict(counts), following={a: dict(b) for a, b in following.items()}, lexicon=sorted(lexicon)), ids
