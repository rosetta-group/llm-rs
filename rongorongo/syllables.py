"""Polynesian text reduced to Rapa Nui-shaped syllable streams.

Following Rochala 2026 (scripts/fetch_polynesian.py): macrons dropped, Māori ``wh`` -> ``h`` and
``w`` -> ``v``, ``ng`` -> ``ŋ``. A word is kept only if it is entirely (consonant) vowel syllables.
"""

import re
import unicodedata

CONSONANTS = "hkmnŋprtv"
VOWELS = "aeiou"
_SYLLABLE = re.compile(rf"[{CONSONANTS}]?[{VOWELS}]")
_WORD = re.compile(rf"(?:[{CONSONANTS}]?[{VOWELS}])+")
ALPHABET = [c + v for c in [""] + list(CONSONANTS) for v in VOWELS]


def normalise_maori(word):
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    w = re.sub(r"[ʻ‘’'`^]", "", w)
    return w.replace("ng", "ŋ").replace("wh", "h").replace("w", "v")


def words(text, normalise=normalise_maori):
    out = []
    for raw in re.split(r"[^A-Za-zÀ-ɏʻ‘’'`^]+", text):
        if raw:
            w = normalise(raw)
            if w and _WORD.fullmatch(w):
                out.append(w)
    return out


def syllables(word_list):
    """One flat stream: word boundaries dropped, as on the tablets."""
    return [s for w in word_list for s in _SYLLABLE.findall(w)]
