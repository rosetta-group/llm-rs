"""Linear A corpus from the pinned Navarre-AI collation (SigLA and lineara.xyz word divisions)."""

import json
from pathlib import Path

from linear_a.spelling import parse_syllabic

CORPUS = Path("artifacts/linear-a-sources/navarre/corpus.json")


def load(path=CORPUS):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def word_tokens(corpus):
    """``(document id, word string)`` for every word division in the corpus."""
    return [(doc, word) for doc, record in corpus.items() for word in (record.get("words") or [])]


def readable_types(corpus, min_syllables=2):
    """Distinct words whose every sign has an assumed Linear B value: ``{syllables: [strings]}``."""
    types = {}
    for _, word in word_tokens(corpus):
        syllables = parse_syllabic(word)
        if syllables is None or len(syllables) < min_syllables:
            continue
        types.setdefault(syllables, set()).add(word)
    return {key: sorted(value) for key, value in types.items()}
